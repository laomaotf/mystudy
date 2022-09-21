import os,sys 
import random
import numpy as np 
from utils.face_description_proxy import  FaceDescriptionProxy
from sklearn.decomposition import PCA
from matplotlib import cm, pyplot as plt 
from tqdm import tqdm


database_dir = "db"


FDProxy = FaceDescriptionProxy()
feats, labels = [],[]
for rdir, _, names in tqdm(os.walk(database_dir)):
    for name in names:
        sname,ext = os.path.splitext(name)
        if ext.lower() not in {".jpg",'.jpeg','.png','.bmp'}:
            continue
        path = os.path.join(rdir,name)
        desc = FDProxy(path)     
        label = path.split(os.path.sep)[-2] 
        feats.append(desc)
        labels.append(label)
        
pca = PCA(n_components=2)
pca.fit(feats)
feats = pca.transform(feats)


colors = {}
for k,label in enumerate(list(set(labels))):
    colors[label] =  k

color_names = ['r','g','b','y','b']

for feat, label in zip(feats,labels):
    c = color_names[colors[label]]
    plt.scatter(x = feat[0], y = feat[1],c=c)
plt.legend()
plt.show()        
 
            