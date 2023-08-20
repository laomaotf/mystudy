import networkx as nx 
import matplotlib.pyplot as plt
import matplotlib
import os,sys,json
from collections import defaultdict
import pandas as pd
matplotlib.rcParams['font.sans-serif'] = "SimHei"

def load_films(indir = "./films/"):
    cache_file = "_{}.cache.csv".format(os.path.basename(__file__))
    if os.path.exists(cache_file):
        films_data = pd.read_csv(cache_file)
        print("load from cache : {}".format(cache_file))
        films = films_data["film"]
        zhuyans = films_data['zhuyan']
        daoyans = films_data['daoyan']
        return [(f,z,d) for (f,z,d) in zip(films,zhuyans, daoyans)]
    
    films_data = []
    files = [os.path.join(indir,f) for f in os.listdir(indir) if os.path.splitext(f)[-1].lower() == '.json']
    for file in files:
        with open(file,"r",encoding='utf-8') as f:
            data = json.load(f)
            zhuyans = data['zhuyan']
            daoyans = data['daoyan']
        film = os.path.splitext(os.path.basename(file))[0]
        for zhuyan in zhuyans:
            for daoyan in daoyans:
                films_data.append((film, zhuyan, daoyan))
                
    df = pd.DataFrame(data={"film":[x[0] for x in films_data],
                            "zhuyan":[x[1] for x in films_data],
                            "daoyan":[x[2] for x in films_data],
                            })
    df.to_csv(cache_file,index=False) 
    return films_data

def list_all_ccs(G):
    return list(nx.connected_components(G))
      
def sort_by_degree(G,reverse=False):
    nd = [(n,d) for n,d in G.degree()] 
    nd = sorted(nd,key=lambda x: x[1],reverse=reverse)
    return nd
        
def visual(data, key_col, value_col,D=100):
    edges = []
    keys = []
    for item in data:
        key,value = item[key_col], item[value_col]
        if len(key) < 1 or len(value) < 1:
            continue
        if key == value: #自导自演
            continue
        keys.append(key)
        edges.append((key,value)) 
    keys = set(list(set(keys)))
    edges = list(set(edges))
    
    G = nx.Graph()
    G.add_edges_from(edges)
    nd = sort_by_degree(G,True)
    keys_selected = []
    for n in nd:
        if n[0] in keys:
            keys_selected.append(n[0])
        if len(keys_selected) >= D:
            break
     
    keys,values = [], []
    kweights, lweights = defaultdict(int),defaultdict(int)
    
    SG = nx.Graph()
    for item in data:
        key,value = item[key_col], item[value_col]
        if key not in keys_selected:
            continue
        if key == value:
            continue
        keys.append(key)
        values.append(value)
        lweights["_".join([key,value])] += 1
        kweights[key] += 1
        SG.add_edge(key,value)
    linkw = [lweights['_'.join([k,v])] for k,v in zip(keys,values)]
    nx.draw(SG,with_labels=True)
    df = pd.DataFrame({"key":keys,"value":values,"weight":linkw})
    df.to_csv("flourish_links.csv",index=False,header=["key","value","weight"]) 
    ukeys,uvalues = list(set(keys)), list(set(values))
    keyw = [kweights[k] + 3 for k in ukeys]
    valuew = [1 for _ in uvalues]
    df = pd.DataFrame({"key":ukeys+uvalues,"weight":keyw+valuew})
    df.to_csv("flourish_points.csv",index=False,header=["key","weight"])
    plt.show()
visual(load_films(),2,1,D=100)
            
    