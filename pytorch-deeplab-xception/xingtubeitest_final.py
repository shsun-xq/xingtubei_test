#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 24 19:26:13 2019

@author: a123
"""
import time
from multiprocessing import freeze_support
import datetime
from xml.dom import minidom
import os
import torch
import cv2
from PIL import Image
import numpy as np
from modeling.deeplab import *
from yltool import *
from torch.utils.data import DataLoader
import sys




def create_xml_test(path, img_id):
    path = path
    img_id = img_id
    
    img_name = '精细化标注_' + ''.join(img_id) + '.tif'
    savepath = path + ''.join(img_id) + '.xml'
    
    xml=minidom.Document()

    root=xml.createElement('Research')

    root.setAttribute('Direction', "高分软件大赛")
    root.setAttribute('ImageName', img_name)

    xml.appendChild(root)
    
    Department = xml.createElement('Department')
    root.appendChild(Department)
    buct = xml.createTextNode('北京化工大学')
    Department.appendChild(buct)
    
    Date = xml.createElement('Date')
    root.appendChild(Date)
    date = xml.createTextNode(str(datetime.date.today()))
    Date.appendChild(date)
    
    PluginName = xml.createElement('PluginName')
    root.appendChild(PluginName)
    pluginname = xml.createTextNode('精细化标注')
    PluginName.appendChild(pluginname)
    
    PluginClass = xml.createElement('PluginClass')
    root.appendChild(PluginClass)
    pluginclass = xml.createTextNode('精细化标注')
    PluginClass.appendChild(pluginclass)
    
    text_node=xml.createElement('Results')
    text_node.setAttribute('Coordinate', "Pixel")
    root.appendChild(text_node)

    result_file = xml.createElement('ResultsFile')
    text_node.appendChild(result_file)
    
    text1 = xml.createTextNode('精细化标注_'+ ''.join(img_id) +'_Results.jpg')
    result_file.appendChild(text1)
    

    f=open(savepath,'wb')
    f.write(xml.toprettyxml(encoding='utf-8'))
    f.close()

def handleImg(img):
    net = DeepLab(backbone='xception', num_classes=9)
    #checkpoint = torch.load(os.path.dirname(os.path.realpath(sys.argv[0])) + '/checkpoint.pth.tar')
    checkpoint = torch.load('./checkpoint.pth.tar')
    
    net.load_state_dict(checkpoint['state_dict'])
    net.eval()
    
    for params in net.parameters():
        params.requires_grad = False
    
    if gpu_flag:
        net = net.cuda()
        img = img.cuda()
        
    _, output = net(img)
    net = 0
    
    output = (output.cpu()).numpy()
    
    return output

def get_color_labels():
    return np.array([[0,128,0], [0,255,0], [0,0,255], [255,255,0], [255,0,255], [0,0,0], [255,255,255], [128,128,128]])
    
def segmap(label_mask):
    label_colours = get_color_labels()
    r = label_mask.copy()
    g = label_mask.copy()
    b = label_mask.copy()
    for ll in range(0, 8):
        r[label_mask == ll] = label_colours[ll, 0]
        g[label_mask == ll] = label_colours[ll, 1]
        b[label_mask == ll] = label_colours[ll, 2]
    rgb = np.zeros((label_mask.shape[0], label_mask.shape[1], 3))
    rgb[:, :, 0] = r
    rgb[:, :, 1] = g
    rgb[:, :, 2] = b
    
    return rgb

class Xingtubei(torch.utils.data.Dataset):
    def __init__(self, root):
        self.img_dir = root
        self.examples = []
        file_names = os.listdir(root)
        for file_name in file_names:
            img_path = root + file_name
            img_id = file_name.split('.')[0]
            example = {}
            example['img_path'] = img_path
            example['img_id'] = img_id
            self.examples.append(example)
        self.num_examples = len(self.examples)
        
    def __getitem__(self, index):
        example = self.examples[index]
        img_id = example['img_id']
        img_path = example['img_path']
        
        if img_path.split('.')[1] == 'tif':
            img = cv2.imread(img_path)
            b,g,r = cv2.split(img)
            img = cv2.merge([r,g,b])
            img = img.astype(np.float32)
        else:
            img = Image.open(img_path)
            img = np.array(img)
            img = img.astype(np.float32)

        
        img = torch.from_numpy(img)
        
        return (img, img_id)
    
    def __len__(self):
        return self.num_examples
    
    
def test():
    for imgs, imgs_id in dataloader:
        start = time.time()
        imgs_ = imgs.permute(1, 2, 0, 3)
        outputs_ = autoSegmentWholeImg(imgs_, (512,512), handleImg, step=300, weightCore='gauss')        
        outputs = outputs_.permute(3, 2, 0, 1)
        outputs = torch.argmax(outputs, dim=1)
        outputs = torch.squeeze(outputs, dim=0)
        outputs = outputs.numpy()
        outputs = segmap(outputs)
        outputs = Image.fromarray(np.uint8(outputs))
        
        name = ''.join(imgs_id)
        outputs.save(savepath + '精细化标注_' + name + '_Results.jpg', quality=100)
        #outputs.save(savepath + '精细化标注_'.join(imgs_id) + '_Results.jpg', quality=100)
        create_xml_test(savepath, '精细化标注_' + name + '_Results')        
        print(''.join(imgs_id) + ' done')
        
        end = time.time()
        print('time: %d s' % (end-start))

    print('done')        
        
freeze_support()

if __name__ == '__main__':
# =============================================================================
#     root = sys.argv[1]
#     savepath = sys.argv[2]
# =============================================================================
    
    root = input('imgPath:')
    savepath = input('resultFile:')
    
    if root[-1] != '/':
        root = root + ''.join('/')
    
    if savepath[-1] != '/':
        savepath = savepath + '/'
    
    xingtubei_test = Xingtubei(root)
    dataloader = DataLoader(xingtubei_test, batch_size=1, num_workers=1, shuffle=False) 
    if torch.cuda.is_available():
        gpu_flag = True
    
    test()
        
        
