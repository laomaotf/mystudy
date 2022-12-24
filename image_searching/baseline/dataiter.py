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

class DATAITER_IMAGE: #each image is one class 
    def __init__(self, imgdir, width = 112, height = 112, K = 32, worker_num = 2, keep_color=False, flag_start_worker = True):
        self.imgpaths = []
        for rdir, _, names in os.walk(imgdir):
            names = list(names)
            names = list(filter(lambda x: os.path.splitext(x)[-1].lower() in {'.jpg','.jpeg','.bmp','.png'}, names))
            self.imgpaths.extend(
                list(map(lambda x: os.path.join(rdir,x),names))
            )
        self.keep_color = keep_color
        if self.keep_color:
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
        else:
            self.strong_transforms = A.SomeOf([
                A.HorizontalFlip(p=1),
                A.VerticalFlip(p=1),
                A.Blur(p=1.0,blur_limit=(5,9)),
                A.RandomBrightnessContrast(p=1),
                A.Affine(p=1,scale=0.9,translate_percent=0.1,rotate=(-10,10)),
                A.GridDistortion(p=1,num_steps=8,distort_limit=0.3),
                A.ChannelShuffle(p=1),
                A.RGBShift(p=1)
            ],3,replace=True)
            self.weak_transforms = A.Compose([
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.Blur(p=0.5,blur_limit=(5,9)),
                A.RandomBrightnessContrast(p=0.5),
                A.Affine(p=0.5,scale=0.9,translate_percent=0.1,rotate=(-10,10)),
                A.GridDistortion(p=0.5,num_steps=8,distort_limit=0.3),
                A.ChannelShuffle(p=0.5),
                A.RGBShift(p=0.5)
            ])
        self.width,self.height = width, height
        self.K = K
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
        paths = random.sample(self.imgpaths,self.K+1) 
        path_pos = paths[0]
        paths_neg = paths[1:]
        
        batch_data = []
        for _ in range(self.K):      
            if self.keep_color:
                mode = 1
            else:
                mode = random.choice([0,1])
            img = cv2.resize(cv2.imread(path_pos,mode),(self.width,self.height),0,0,interpolation=cv2.INTER_AREA)
            if len(img.shape) == 2:
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)            
            img = self.do_augment(img,self.strong_transforms)
            img = np.expand_dims(img.transpose(2,0,1),axis=0)
            batch_data.append( img )
       
        for k in range(self.K):    
            if self.keep_color:
                mode = 1
            else:
                mode = random.choice([0,1])  
            img = cv2.resize(cv2.imread(paths_neg[k],mode),(self.width,self.height),0,0,interpolation=cv2.INTER_AREA)
            if len(img.shape) == 2:
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)            
            img = self.do_augment(img,self.weak_transforms)
            img = np.expand_dims(img.transpose(2,0,1),axis=0)
            batch_data.append( img )
        
        return torch.from_numpy((np.concatenate(batch_data,axis=0) - 128.0)/128.0)
            
    def next(self):
        if self.batches.empty():
            logging.warning("empty queue")
        return self.batches.get(block=True)



class DATAITER_CLASS(DATAITER_IMAGE): #use folder name as label
    def __init__(self, imgdir, width=112, height=112, K=32, worker_num=2):
        super().__init__(imgdir, width, height, K, worker_num,flag_start_worker=False) 
        self.imgpaths_class = defaultdict(list)
        for path in self.imgpaths:
            c = path.split(os.path.sep)[-2]
            self.imgpaths_class[c].append(path)
        self.start_worker(True) 
        return
    def get_one_batch(self):
        class_pos,class_neg = random.sample(self.imgpaths_class.keys(),2) 
        paths_pos = random.choices(self.imgpaths_class[class_pos],k = self.K)
        paths_neg = random.choices(self.imgpaths_class[class_neg],k = self.K)
        
        batch_data = []
        for path in paths_pos:    
            if self.keep_color:
                mode = 1
            else:
                mode = random.choice([0,1])                
            img = cv2.resize(cv2.imread(path,mode),(self.width,self.height),0,0,interpolation=cv2.INTER_AREA)
            if len(img.shape) == 2:
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            img = self.do_augment(img,self.weak_transforms)
            img = np.expand_dims(img.transpose(2,0,1),axis=0)
            batch_data.append( img )
        for path in paths_neg:      
            if self.keep_color:
                mode = 1
            else:
                mode = random.choice([0,1])               
            img = cv2.resize(cv2.imread(path,mode),(self.width,self.height),0,0,interpolation=cv2.INTER_AREA)
            if len(img.shape) == 2:
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)            
            img = self.do_augment(img,self.weak_transforms)
            img = np.expand_dims(img.transpose(2,0,1),axis=0)
            batch_data.append( img )
        batch_data = (np.concatenate(batch_data,axis=0) - 128.0) / 128.0
        return torch.from_numpy(batch_data)       

          
def testbed_dataiter():
    batches = DATAITER_IMAGE("/dataset/train",worker_num=1)
    for index in range(2):
        batch_data = np.clip(batches.next().numpy() * 128  + 127,0,255).astype(np.uint8)
        K = batch_data.shape[0] //2
        img_anchor, img_pos = batch_data[0], batch_data[1]
        img_anchor = img_anchor.transpose(1,2,0) 
        img_pos = img_pos.transpose(1,2,0)
        d0 = np.hstack((img_anchor, img_pos))
        
        img_anchor, img_neg = batch_data[0], batch_data[K]
        img_anchor = img_anchor.transpose(1,2,0)
        img_neg = img_neg.transpose(1,2,0)
        d1 = np.hstack((img_anchor, img_neg))
        cv2.imshow("vis",np.vstack([d0,d1]))
        cv2.waitKey(2000)
        print(index)
    del batches
    
if __name__=="__main__":
    testbed_dataiter() 