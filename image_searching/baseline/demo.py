import shutil
import torch
import pickle
import os,sys
import numpy as np
from network import NETWORK
import cv2
from rich.progress import track
from collections import defaultdict

class FEATDB:
    def __init__(self, model_file,width,height, db_file="",device="cuda"):
        self.network, self.db = None, None
        self.width,self.height = width,height
        self.device = device
        self.network = NETWORK()
        self.network.load_state_dict(torch.load(model_file))
        self.network.eval()
        self.network.to(device)
        if db_file != "" and os.path.exists(db_file):
            with open(db_file,'rb') as f:
                self.db = pickle.load(f)
        else:
            self.db = {
                "path":defaultdict(int),
                "feat":None
            }
        return
    def _generate(self,path):
        if path in self.db['path'].keys():
            n = self.db['path'][path]
            return self.db['feat'][n]
        img = cv2.imread(path,1)
        if len(img.shape) == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        img = cv2.resize(img,(self.width,self.height),0,0,cv2.INTER_AREA)
        img = (img.astype(np.float32) - 128.0)/128.0
        img = np.expand_dims(np.transpose(img,(2,0,1)),0)
        
        data = torch.from_numpy(img).to(self.device)
        feat = self.network(data).detach().cpu().numpy()
        return np.reshape(feat,(1,-1))

    def _insert_or_update(self,path,feat,flag_update=True):
        if self.db['feat'] is None:
            self.db['path'][path] = 0
            self.db['feat'] = feat
        else:
            if path not in self.db['path']:
                self.db['path'][path] = self.db['feat'].shape[0] 
                self.db['feat'] = np.vstack((self.db['feat'],feat))
            elif flag_update:
                row = self.db['path'][path]
                self.db['feat'][row] = feat
        return 
    def add(self,img_dir,**kwargs):
        paths_all = []
        for rdir, _, names in os.walk(img_dir):
            for name in names:
                _, ext = os.path.splitext(name)
                if ext.lower() not in {".jpg",".png"}:
                    continue
                path = os.path.join(rdir,name)
                paths_all.append(path)
        for path in track(paths_all):
            feat = self._generate(path)
            self._insert_or_update(path,feat,**kwargs) 
        return 
    def save(self,path):
        with open(path,'wb') as f:
            pickle.dump(self.db,f)
        return
    def _get_path(self,row_index):
        paths = list(filter(lambda key: self.db['path'][key] == row_index,self.db['path'].keys()))
        return paths
        
    def search(self,query_file,outdir,topk = 9):
        os.makedirs(outdir,exist_ok=True)
        feat = self._generate(query_file)
        dist = np.linalg.norm(self.db['feat'] - feat,axis=-1,ord=2)
        argrows = np.argsort(dist) 
        paths_found = []
        for k in range(min(topk,len(argrows))):
            paths = self._get_path(argrows[k])
            paths_found.extend(paths)
        for k,path in enumerate(paths_found):
            cls = os.path.split(os.path.split(path)[0])[-1]
            outpath = os.path.join(outdir, f"{k}_{cls}_{os.path.basename(path)}")
            shutil.copy(path,outpath)
        return
    
    
if __name__ == "__main__":
    model_path = os.path.join(os.path.dirname(__file__),"run/step_113000.pth")
    featdb = FEATDB(model_path,96,96,db_file="valtest.pkl")
    featdb.add("/dataset/test")
    featdb.save("valtest.pkl")
    featdb.search("/dataset/sister_one.png","same_one",topk=50)
    
        
            
            
            
        
        
        
         
        
                        
                

