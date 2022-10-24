import cv2
from scipy import rand
import torch
import numpy as np
import os,sys
import logging
import random
import multiprocessing as mp
import albumentations as A
from collections import defaultdict

class DATAITER: #each image is one class 
    def __init__(self, imgdir, width = 112, height = 112, batch_size = 32, worker_num = 2, flag_start_worker = True):
        self.imgpaths = []
        for rdir, _, names in os.walk(imgdir):
            names = list(names)
            names = list(filter(lambda x: os.path.splitext(x)[-1].lower() in {'.jpg','.jpeg','.bmp','.png'}, names))
            self.imgpaths.extend(
                list(map(lambda x: os.path.join(rdir,x),names))
            )
            
        self.strong_transforms = A.SomeOf([
            A.HorizontalFlip(p=1),
            A.VerticalFlip(p=1),
            A.Blur(p=1.0,blur_limit=(5,9)),
            A.RandomBrightnessContrast(p=1),
            A.Affine(p=1,scale=0.9,translate_percent=0.1,rotate=(-10,10)),
            A.GridDistortion(p=1,num_steps=8,distort_limit=0.3),
        ],3,replace=True)
        self.weak_transforms = A.Compose([
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.Blur(p=0.5,blur_limit=(5,9)),
            A.RandomBrightnessContrast(p=0.5),
            A.Affine(p=0.5,scale=0.9,translate_percent=0.1,rotate=(-10,10)),
            A.GridDistortion(p=0.5,num_steps=8,distort_limit=0.3),
        ])
        
        
        self.width,self.height = width, height
        self.batch_size = batch_size
        self.worker_num = 1 if worker_num < 1 else worker_num
        self.batches = mp.Queue(maxsize=worker_num*5)
        self.workers = [mp.Process(target=self.do_jobs) for _ in range(worker_num)]
        self.start_worker(flag_start_worker)
        return 
    def start_worker(self,flag):
        if not flag:
            return 
        for worker in self.workers:
            worker.daemon = True
            worker.start()        
        return  
    def do_jobs(self):
        while True:
            batch_data = self.get_one_batch()
            self.batches.put(batch_data,block=True)
        
        
    def do_augment(self, img, transforms):
        img_aug = transforms(image=img)["image"]
        return img_aug
    
    def get_one_batch(self):  
        batch_data = []
        for _ in range(self.batch_size):      
            path_anchor,path_neg = random.sample(self.imgpaths,2) 
            img_anchor = cv2.resize(cv2.imread(path_anchor,1),(self.width,self.height),0,0,interpolation=cv2.INTER_AREA)
            img_pos = self.do_augment(img_anchor,self.strong_transforms)
            img_neg = cv2.resize(cv2.imread(path_neg,1),(self.width,self.height),0,0,interpolation=cv2.INTER_AREA)
            
            img_anchor = np.expand_dims(img_anchor.transpose(2,0,1),axis=0)
            img_pos = np.expand_dims(img_pos.transpose(2,0,1),axis=0)
            img_neg = np.expand_dims(img_neg.transpose(2,0,1),axis=0)
            batch_data.append( np.concatenate([img_anchor, img_pos, img_neg],axis=0) )
        return torch.from_numpy((np.concatenate(batch_data,axis=0) - 128.0)/128.0)
            
    def next(self):
        if self.batches.empty():
            logging.warning("empty queue")
        return self.batches.get(block=True)

class DATAITER_CLASS(DATAITER): #use folder name as label
    def __init__(self, imgdir, width=112, height=112, batch_size=32, worker_num=2):
        super().__init__(imgdir, width, height, batch_size, worker_num,flag_start_worker=False) 
        self.imgpaths_class = defaultdict(list)
        for path in self.imgpaths:
            c = path.split(os.path.sep)[-2]
            self.imgpaths_class[c].append(path)
        self.start_worker(True) 
        return
    def get_one_batch(self):
        batch_data = []
        for _ in range(self.batch_size):      
            class_anchor,class_neg = random.sample(self.imgpaths_class.keys(),2) 
            path_anchor, path_pos = random.sample(self.imgpaths_class[class_anchor],2)
            path_neg = random.sample(self.imgpaths_class[class_neg],1)[0]
            
            img_anchor = cv2.resize(cv2.imread(path_anchor,1),(self.width,self.height),0,0,interpolation=cv2.INTER_AREA)
            img_pos = cv2.resize(cv2.imread(path_pos,1),(self.width,self.height),0,0,interpolation=cv2.INTER_AREA)
            img_pos = self.do_augment(img_pos,self.weak_transforms)
            img_neg = cv2.resize(cv2.imread(path_neg,1),(self.width,self.height),0,0,interpolation=cv2.INTER_AREA)
            
            img_anchor = np.expand_dims(img_anchor.transpose(2,0,1),axis=0)
            img_pos = np.expand_dims(img_pos.transpose(2,0,1),axis=0)
            img_neg = np.expand_dims(img_neg.transpose(2,0,1),axis=0)
            batch_data.append( np.concatenate([img_anchor, img_pos, img_neg],axis=0) )
        return torch.from_numpy((np.concatenate(batch_data,axis=0) - 128.0)/128.0)        
    
          
def testbed_dataiter():
    batches = DATAITER_CLASS(r'val')
    for index in range(10):
        batch_data = np.clip(batches.next().numpy() * 128  + 127,0,255).astype(np.uint8)
        img_anchor, img_pos, img_neg = batch_data[0], batch_data[1],batch_data[2]
        img_anchor = img_anchor.transpose(1,2,0) 
        img_pos = img_pos.transpose(1,2,0)
        img_neg = img_neg.transpose(1,2,0)
        d0 = np.hstack((img_anchor, img_pos, img_neg))
        
        img_anchor, img_pos, img_neg = batch_data[3], batch_data[4],batch_data[5]
        img_anchor = img_anchor.transpose(1,2,0)
        img_pos = img_pos.transpose(1,2,0)
        img_neg = img_neg.transpose(1,2,0)
        d1 = np.hstack((img_anchor, img_pos, img_neg))
        cv2.imshow("vis",np.vstack([d0,d1]))
        cv2.waitKey(2000)
        print(index)
    del batches
if __name__=="__main__":
    testbed_dataiter() 