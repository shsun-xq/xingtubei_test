import numpy as np
import torch

def getWeightCore(hh,ww=None,mappFun=None,seta=0.5):
    '''
    返回一个权重二维矩阵 ，默认是seta=0.5的二维高斯分布
    mappFun: 对矩阵每个点执行mappFun(i,j) 返回的结果构成二维矩阵
    '''
    if ww is None:
        ww = hh
    if mappFun is None:
#        ijToCenter = lambda x,i,j:(((i/float(hh)-1/2.)**2+(j/float(ww)-1/2.)**2))
#        wc = weightCore = mapp(ijToCenter,weightCore,need_i_j=True)
        i,j = np.mgrid[:hh,:ww]
        wc = (((i/float(hh)-1/2.)**2+(j/float(ww)-1/2.)**2))
        wc = 1./(2*np.pi*seta**2)*np.e**(-wc/(2*seta**2))
        wc = wc/wc.max()
        #show(normalizing(img[:hh,:ww]*wc[...,None]),img[:hh,:ww])
#        polt3dSurface(wc)
        return wc
    weightCore = np.zeros((hh,ww))
    return mapp(lambda x,i,j:mappFun(i,j),weightCore,need_i_j=True)

def smallImg(img,simgShape, step=None,f=None):
    '''
    将大图切割成固定大小的小图,使产生的小图能覆盖大图的所有面积
    Parameters
    ----------
    simgShape : int or tuple or float
        小图片的shape,为int时候 自动转换为(simgShape, simgShape)
        为float时候 转换为 (int(h*simgShape),int(w*simgShape))
    step : float or int or tuple(steph,stepw),defalut None
        h和w方向上 相邻切割的步长, 默认为simgShape 
        float : (int(step*simgShape[0]),int(step*simgShape[1]))
        int : (step, step)
    fun : funcatin, default None
        若有fun 则执行fun(simg,i,j)
        其中：
            simg:被切割的小图片
            i: simg所在img的row
            j: simg所在img的col
    
    Returns
    -------
    simgs : list of ndarray
        切割出来的小图的list
    '''
    h,w = img.shape[:2]
    if isinstance(simgShape,float):
        hh,ww = (int(h*simgShape),int(w*simgShape))
    if isinstance(simgShape,int):
        hh,ww = simgShape,simgShape
    if isinstance(simgShape,(tuple,list)):
        hh,ww = simgShape
    if step is None:
        steph,stepw = hh,ww
    if isinstance(step,int):
        steph,stepw = step,step
    if isinstance(step,float):
        steph,stepw = int(hh*step),int(ww*step)
    if isinstance(step,(tuple,list)):
        steph,stepw = step
    simgs = []
    py3_ls = np.arange(0,h-hh,steph)
    py3_ls = np.concatenate((py3_ls, [h-hh]))
   #for i in range(0,h-hh,steph)[:]+[h-hh]:
    for i in py3_ls:
        py3_ls1 = np.arange(0,w-ww,stepw)
        py3_ls1 = np.concatenate((py3_ls1, [w-ww]))
      #  for j in range(0,w-ww,stepw)[:]+[w-ww]:
        for j in py3_ls1:
            simg = img[i:i+hh,j:j+ww]
            simgs.append(simg)
            if f:
                f(simg,i,j)
    return simgs


def autoSegmentWholeImg(img,simgShape,handleSimg,step=None,weightCore=None):
    '''
    将img分割到 shape为simgShape 的小图simg，执行handleSimg(simg)
    将所有handleSimg(simg)的结果自动拼接成img形状的ndarray并返回
    
    Parameters
    ----------
    img : ndarray
        需要被分割处理的图片
    simgShape : int or tuple
        小图片的shape,为int时候 自动转换为(simgShape, simgShape)
    handleSimg : funcation
        用于处理shape为simgShape的小图片的函数 
        此函数需要接受一个ndarray作为参数并返回shape[:2]同为为(h,w)的ndarray
        即：handleSimg(simg)=>ndarray，比如 net.pridict(simg)
    step : float or int or tuple(steph,stepw),defalut None
        h和w方向上 相邻切割的步长, 默认为simgShape 
        float : (int(step*simgShape[0]),int(step*simgShape[1]))
        int : (step, step)
    weightCore : {None,'avg','gauss',ndarray}, defalut None 
        对于两个simg图片重叠部分进行融合时候的各自权重
        默认取距离simg中心最近的部分
       'gauss':在重叠部分 采用高斯分布 使之离simg中心越远，权重越低
       'avg':重叠部分取平均
    
    Returns
    -------
    result : ndarray
        shape[:2]等于img.shape[:2]的ndarray
    '''
    if isinstance(simgShape,int):
        hh,ww = simgShape,simgShape
    hh,ww = simgShape
    h,w = img.shape[:2]
    if weightCore is None:
        pass
    elif isinstance(weightCore,np.ndarray):
        pass
    elif weightCore in ['avg']:
        weightCore = np.ones((hh,ww))
    elif weightCore in ['guss','gauss']:
        weightCore = getWeightCore(hh,ww)
    else:
        raise Exception('Illegal argus `weightCore` in `autoSegmentWholeImg`!')
    weight = np.zeros((h,w))
    class c:
        re=None
        near=None
    def f(simg,i,j):
        simg = simg.permute(2, 3, 0, 1)
        sre = handleSimg(simg)
        #sre = sre.permute(2, 3, 1, 0)
        #print(sre.shape)
        #print(sre.dtype)
        sre = sre.transpose(2, 3, 1, 0)
        if c.re is None:
            c.re = np.zeros((h,w)+sre.shape[2:],sre.dtype)
        if weightCore is None:
            if c.near is None:
                y,x = np.mgrid[:hh,:ww]
                c.near = 1-((x*1./ww-1./2)**2+(y*1./hh-1./2)**2)**.5
            ind = c.near > weight[i:i+hh,j:j+ww]
            c.re[i:i+hh,j:j+ww][ind]= sre[ind]
            weight[i:i+hh,j:j+ww][ind]= c.near[ind]
            return
        oldw = weight[i:i+hh,j:j+ww]
        ws = weightCore
        if sre.ndim!=2:
            ws = ws[...,None, None]
            oldw = oldw[...,None, None]
    #    map(loga,[ws,sre,c.re,oldw,c.re[i:i+hh,j:j+ww]*oldw])
        c.re[i:i+hh,j:j+ww] = (ws*sre + c.re[i:i+hh,j:j+ww]*oldw)/(ws+oldw)
        weight[i:i+hh,j:j+ww] += weightCore
    #    show(c.re,weight)
    (smallImg(img,(hh,ww),step=step,f=f))
    c.re = torch.from_numpy(c.re)
    return c.re

from PIL import Image
if __name__ == '__main__':
    
    PIL_img = Image.open('/home/lmming/lmming/code/dataset/HRF/train/images/08_g.jpg')
    np_img  = np.array(PIL_img)
    res     = autoSegmentWholeImg(np_img, (640, 640), smallImg) 
