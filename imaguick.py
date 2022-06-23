import torch.utils.data as data
from PIL import Image, ImageDraw, ImageOps
import torch
import torch.nn as nn
from collections import Counter

#from scipy.misc.pilutil import imread
#from matplotlib.pyplot import imread
from cv2 import imread
from torchvision.utils import save_image
import torchvision.transforms as transforms
import numpy as np
import cv2
import random
import time
import pdb
import pickle
import random
from collections import defaultdict
import os
import json
import ndjson

def convert_to_np_raw(drawing, width=224, height=224):
    img = np.zeros((width, height))
    pil_img = convert_to_PIL(drawing)
    pil_img.thumbnail((width, height), Image.ANTIALIAS)
    pil_img = pil_img.convert('RGB')
    pixels = pil_img.load()

    for i in range(0, width):
        for j in range(0, height):
            img[i,j] = 1- pixels[j,i][0]/255.0
    # return img
    return pil_img

def convert_to_PIL(drawing, width=224, height=224): # 256 before
    pil_img = Image.new('RGB', (width, height), 'white')
    pixels = pil_img.load()
    draw = ImageDraw.Draw(pil_img)
    for x,y in drawing:
        for i in range(1, len(x)):
            draw.line((x[i-1], y[i-1], x[i], y[i]), fill=0)
    return pil_img

def normalize_transform():
    return transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

class ImagenetQuickdraw(data.Dataset):
    def __init__(self, imagenet_path, quickdraw_path,quick2image, class_path, train=True):
        _imagenet_path = imagenet_path
        _quickdraw_path = quickdraw_path
        self.quick2image = quick2image
        self.class_path = class_path

        _quickdraw_path = pickle.load(open(_quickdraw_path, 'rb'))
        print("Loading Quick,Draw! ...")
        if train:
            _quickdraw_path = _quickdraw_path['train_x']
        else:
            _quickdraw_path = _quickdraw_path['valid_x']

        self.quick2image = open(self.quick2image, 'r').readlines()
        self.class_path = open(self.class_path, 'r').readlines()
        self.image2quick = {}

        _id2label = open('/home/pnoel/mdetr/id2label.txt', 'r').readlines()
        _label2idx = {}
        self.all_classes = []
        for data in _id2label:
            data_id, data_label = data.strip()[0:-1].split(':')
            data_label = data_label.strip()[1:-1].split(',')
            for dl in data_label:
                _label2idx[dl.strip().replace(' ', '_')] = data_id.strip()

        

        for data in self.quick2image:
            quick, image = data.split(':')
            image = image.split(',')
            for i in image:
                self.image2quick[i.strip()] = quick[1:-1]
                self.all_classes.append(quick[1:-1])
        self.all_classes = list(set(self.all_classes))
        self.class2folder = {}
        for data in self.class_path:
            folder, _, name = data.strip().split(' ')
            self.class2folder[_label2idx[name]] = folder

        self.all_images = []

        for class_id, quickdraw in  self.image2quick.items():
            class_folder = self.class2folder[class_id]
            all_files = os.listdir(os.path.join(_imagenet_path, class_folder))
            for file in all_files:
                file_path = os.path.join(os.path.join(_imagenet_path, class_folder), file)
                self.all_images.append((file_path, quickdraw))

        self.class2quick = defaultdict(list)
        # for cat in self.all_classes:
        #         cat_file = ndjson.load(open(os.path.join(_quickdraw_path, cat+'.ndjson')))
        #         print(cat_file.keys())
        #         aditay
        for path in _quickdraw_path:
            cat = path.split('/')[-2]
            self.class2quick[cat].append(path)
        if train:
            self.train_transforms = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize_transform()
            ])

            self.train_transforms_sketch = transforms.Compose([
            # transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize_transform()
            ])

        else:
            self.train_transforms = transforms.Compose([
            transforms.RandomResizedCrop(224),
            # transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize_transform()
            ])

            self.train_transforms_sketch = transforms.Compose([
            # transforms.RandomResizedCrop(224),
            # transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize_transform()
            ])

        print("Number of Images: %d"%(len(self.all_images)))
        self.train = train

    def __len__(self):
        return len(self.all_images)
        # return 400

    def __getitem__(self, index):
        image, cat = self.all_images[index]
        # print(len(self.all_images))
        
        pos_cat = cat
        neg_cat = random.choice(list(set(self.all_classes) - set([pos_cat])))
        
        positive_sketch = random.choice(self.class2quick[pos_cat])
        negative_sketch = random.choice(self.class2quick[neg_cat])

        positive_sketch = pickle.load(open(positive_sketch, 'rb'))
        negative_sketch = pickle.load(open(negative_sketch, 'rb'))
        
        key = list(positive_sketch.keys())[0]
        positive_sketch = convert_to_np_raw(positive_sketch[key])

        key = list(negative_sketch.keys())[0]
        negative_sketch = convert_to_np_raw(negative_sketch[key])

        # positive_sketch = np.stack((positive_sketch, positive_sketch, positive_sketch), axis=0)

        # negative_sketch = np.stack((negative_sketch, negative_sketch, negative_sketch), axis=0)

        # print(positive_sketch.shape, negative_sketch.shape)
        # positive_sketch.save("pos.jpg")
        # negative_sketch.save("neg.jpg")

        img = Image.open(image).convert("RGB")
        # img.save("orig.jpg")
        # aditay
        # if self.train:
        img = self.train_transforms(img)
        # positive_sketch = Image.fromarray(np.uint8(positive_sketch)).convert('RGB')
        positive_sketch = self.train_transforms_sketch(positive_sketch)
        # negative_sketch = Image.fromarray(np.uint8(negative_sketch)).convert('RGB')
        negative_sketch = self.train_transforms_sketch(negative_sketch)
        
        # positive_sketch = torch.from_numpy(positive_sketch)
        # negative_sketch = torch.from_numpy(negative_sketch)
        data = {}
        data['samples'] = img
        data['pos_samples'] = positive_sketch
        data['neg_samples'] = negative_sketch
        return img, positive_sketch, negative_sketch

class ImagenetQuickdrawVal(data.Dataset):
    def __init__(self, imagenet_path, quickdraw_path,quick2image, class_path, train=False):
        _imagenet_path = imagenet_path
        _quickdraw_path = quickdraw_path
        self.quick2image = quick2image
        self.class_path = class_path

        _quickdraw_path = pickle.load(open(_quickdraw_path, 'rb'))
        print("Loading Quick,Draw! ...")
        # if train:
        if False:
            _quickdraw_path = _quickdraw_path['train_x']
        else:
            _quickdraw_path = _quickdraw_path['valid_x']

        self.quick2image = open(self.quick2image, 'r').readlines()
        self.class_path = open(self.class_path, 'r').readlines()
        self.image2quick = {}

        _id2label = open('/home/pnoel/mdetr/id2label.txt', 'r').readlines()
        _label2idx = {}
        self.all_classes = []
        for data in _id2label:
            data_id, data_label = data.strip()[0:-1].split(':')
            data_label = data_label.strip()[1:-1].split(',')
            for dl in data_label:
                _label2idx[dl.strip().replace(' ', '_')] = data_id.strip()

        

        for data in self.quick2image:
            quick, image = data.split(':')
            image = image.split(',')
            for i in image:
                self.image2quick[i.strip()] = quick[1:-1]
                self.all_classes.append(quick[1:-1])
        self.all_classes = list(set(self.all_classes))
        self.class2folder = {}
        for data in self.class_path:
            folder, _, name = data.strip().split(' ')
            self.class2folder[_label2idx[name]] = folder

        self.all_images = []

        for class_id, quickdraw in  self.image2quick.items():
            class_folder = self.class2folder[class_id]
            all_files = os.listdir(os.path.join(_imagenet_path, class_folder))
            for file in all_files:
                file_path = os.path.join(os.path.join(_imagenet_path, class_folder), file)
                self.all_images.append((file_path, quickdraw))

        self.class2quick = defaultdict(list)
        # for cat in self.all_classes:
        #         cat_file = ndjson.load(open(os.path.join(_quickdraw_path, cat+'.ndjson')))
        #         print(cat_file.keys())
        #         aditay
        for path in _quickdraw_path:
            cat = path.split('/')[-2]
            self.class2quick[cat].append(path)
        # if train:
        if False:
            self.train_transforms = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize_transform()
            ])

            self.train_transforms_sketch = transforms.Compose([
            # transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize_transform()
            ])

        else:
            self.train_transforms = transforms.Compose([
            transforms.RandomResizedCrop(224),
            # transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize_transform()
            ])

            self.train_transforms_sketch = transforms.Compose([
            # transforms.RandomResizedCrop(224),
            # transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize_transform()
            ])

        print("Number of Images: %d"%(len(self.all_images)))
        self.train = train

    def __len__(self):
        
        return len(self.all_images)
        # return 400
        

    def __getitem__(self, index):
        image, cat = self.all_images[index]
        
        
        pos_cat = cat
        neg_cat = random.choice(list(set(self.all_classes) - set([pos_cat])))
        
        positive_sketch = random.choice(self.class2quick[pos_cat])
        negative_sketch = random.choice(self.class2quick[neg_cat])

        positive_sketch = pickle.load(open(positive_sketch, 'rb'))
        negative_sketch = pickle.load(open(negative_sketch, 'rb'))
        
        key = list(positive_sketch.keys())[0]
        positive_sketch = convert_to_np_raw(positive_sketch[key])

        key = list(negative_sketch.keys())[0]
        negative_sketch = convert_to_np_raw(negative_sketch[key])

        # positive_sketch = np.stack((positive_sketch, positive_sketch, positive_sketch), axis=0)

        # negative_sketch = np.stack((negative_sketch, negative_sketch, negative_sketch), axis=0)

        # print(positive_sketch.shape, negative_sketch.shape)
        # positive_sketch.save("pos.jpg")
        # negative_sketch.save("neg.jpg")

        img = Image.open(image).convert("RGB")
        # img.save("orig.jpg")
        # aditay
        # if self.train:
        img = self.train_transforms(img)
        # positive_sketch = Image.fromarray(np.uint8(positive_sketch)).convert('RGB')
        positive_sketch = self.train_transforms_sketch(positive_sketch)
        # negative_sketch = Image.fromarray(np.uint8(negative_sketch)).convert('RGB')
        negative_sketch = self.train_transforms_sketch(negative_sketch)
        
        # positive_sketch = torch.from_numpy(positive_sketch)
        # negative_sketch = torch.from_numpy(negative_sketch)
        data = {}
        data['samples'] = img
        data['pos_samples'] = positive_sketch
        data['neg_samples'] = negative_sketch
        return img, positive_sketch, negative_sketch
