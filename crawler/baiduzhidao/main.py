
import requests
from bs4 import BeautifulSoup
import re
import json
import os
import time
from tqdm.auto import tqdm
import random
from collections import defaultdict

ALLOW_DOWNLOAD = False

htmls_dir = "htmls"
celebrities_dir = "celebrities"
films_dir = "films"
director_dir = "directors"

os.makedirs(htmls_dir,exist_ok=True)
os.makedirs(celebrities_dir,exist_ok=True)
os.makedirs(films_dir,exist_ok=True)
os.makedirs(director_dir,exist_ok=True)
with open("common.json",'r') as f:
    common_data = json.load(f)
    
def make_header():
    return {
    "User-Agent":common_data['headers']['User-Agent'],
    "Accept": common_data['headers']['Accept'],
    "Accept-Language": common_data['headers']['Accept-Language']}
    
def make_valid_filename(name):
    for bad_c in ['/','\\']:
        name = name.replace(bad_c,'-')
    return name.strip()
        
class BaikeCrawler:
    def __init__(self,name) -> None:
        self.name = name
        self.url_base = "https://baike.baidu.com/item/"
    def getPage(self, url):
        text = ""
        html_path = os.path.join("{}/{}.html".format(htmls_dir,make_valid_filename(self.name)))
        try:
            if not os.path.exists(html_path):
                if not ALLOW_DOWNLOAD:
                    return None,None
                print('download ',self.name)
                time.sleep(60 + int(random.random() * 30))
                text = requests.get(url,headers=make_header()).text
                with open(html_path,'w',encoding="utf-8",errors="strict") as f:
                    f.write(text)
            else:
                with open(html_path,'r',encoding='utf-8') as f:
                    text = f.read()
        except requests.exceptions.RequestException:
            return None,None
        return BeautifulSoup(text, "html.parser"),text
 

class Celebrity(BaikeCrawler):
    def __init__(self,name) -> None:
        super().__init__(name)
        return
    def get_films(self):
        bs,page_text = self.getPage(self.url_base + self.name + "?")
        films = []
        if bs is not None:
            links = bs.find_all("a",recursive=True)
            #films = re.findall(r"[\u4e00-\u9fa5_0-9a-zA-Z]*</a></b><b>",page_text)
            for link in links:
                linktext = link.text.strip()
                if linktext == "" or len(linktext) > 50:
                    continue
                try:
                    err_text = False
                    for sym in ["《",'|']:
                        if linktext.find(sym) >= 0:
                            err_text = True
                            break
                    if err_text:
                        continue
                    if re.findall(f">{linktext}</a></b><b>",page_text):
                        films.append(linktext) 
                except Exception as e:
                    print(e)
            
        #films = [f.encode("utf-8").decode('utf-8') for f in sorted(list(set(films)))]
        films = sorted(list(set(films)))
        with open(os.path.join("{}/{}.json".format(celebrities_dir,make_valid_filename(self.name))),"w",encoding='utf-8') as f:
            json.dump({"films":films},f,indent=4,ensure_ascii=False)
        return 

def parse_date(node):
    node_dd = [dd for dd in node.next_siblings if dd.name == "dd"][0] #next_siblings: next of tag
    date_text = node_dd.text
    return date_text.strip()

def min_date(dates):
    if dates == []:
        return '00000000'
    normalized = []
    for d in dates:
        yy,mm,dd = "0000","00","00"
        if re.match(r"[\d]{4}年[\d]+月[\d]+日",d):
            yy,mm,dd = re.findall(r"([\d]{4})年([\d]+)月([\d]+)日",d)[0]
            mm = "0"+mm if len(mm) == 1 else mm
            dd = '0'+dd if len(dd) == 1 else dd
        elif re.match(r"[\d]{4}年[\d]+月",d):
            yy,mm = re.findall(r"([\d]{4})年([\d]+)月",d)[0]
            mm = "0"+mm if len(mm) == 1 else mm
            dd = '00'
        elif re.match(r"[\d]{4}年",d):
            yy = re.findall(r"([\d]{4})年",d)[0]
            mm = '00'
            dd = '00'
        normalized.append(yy+mm+dd)
            
    return min(normalized)
            
    
def parse_names_with_link(node):
    names = []
    node_dd = [dd for dd in node.next_siblings if dd.name == "dd"][0] #next_siblings: next of tag
    nodes_a = [a for a in node_dd.children if a.name == 'a'] # children: children of tag
    for name_node in nodes_a: 
        name = name_node.text.strip()
        if len(name) < 1:
            continue
        names.append(name_node.text.strip())
    nodes_others = [a for a in node_dd.children if a.name != 'a']
    for name_node in nodes_others:
        name = name_node.text.strip()
        for sep in [' ']:
            name = name.replace(sep,'')
        for sep in ['、','，']:
            name = name.replace(sep,',')
        name = name.strip().strip(',')
        if len(name) < 1:
            continue
        if re.match(r"\[[\d]+\]",name):
            continue
        if name.find(',') >= 0:
            name = [n.strip() for n in name.split(',')]
        else:
            name = [name]
        if len(name) < 1:
            continue
        names.extend(name)
    return names

class Film(BaikeCrawler):
    def __init__(self, name) -> None:
        super().__init__(name)
        return
    def get_info(self):
        """
<dt class="basicInfo-item name" id="basic-name">导&nbsp;&nbsp;&nbsp;&nbsp;演</dt>
<dd class="basicInfo-item value">
<a target=_blank href="/item/%E5%85%B3%E6%B0%B8%E5%BF%A0/4350897" data-lemmaid="4350897">关永忠</a>
</dd>
<dt class="basicInfo-item name" id="basic-name">编&nbsp;&nbsp;&nbsp;&nbsp;剧</dt>
<dd class="basicInfo-item value">
<a target=_blank href="/item/%E5%8F%B6%E4%B8%96%E5%BA%B7/10923560" data-lemmaid="10923560">叶世康</a>、<a target=_blank href="/item/%E5%AD%99%E6%B5%A9%E6%B5%A9/943485" data-lemmaid="943485">孙浩浩</a>
</dd>
<dt class="basicInfo-item name" id="basic-name">主&nbsp;&nbsp;&nbsp;&nbsp;演</dt>
<dd class="basicInfo-item value">
<a target=_blank href="/item/%E6%B1%AA%E6%98%8E%E8%8D%83/1469785" data-lemmaid="1469785">汪明荃</a>、<a target=_blank href="/item/%E9%A9%AC%E5%BE%B7%E9%92%9F/2506700" data-lemmaid="2506700">马德钟</a>、<a target=_blank href="/item/%E4%BD%98%E8%AF%97%E6%9B%BC/338801" data-lemmaid="338801">佘诗曼</a>
</dd>
        """
        saved_path = os.path.join("{}/{}.json".format(films_dir,make_valid_filename(self.name)))
        if os.path.exists(saved_path):
            return
        zhuyan_all,daoyan_all,dates_all = [],[],[]
        bs,page_text = self.getPage(self.url_base + self.name + "?")
        if bs is not None:
            nodes = bs.find_all("dt",{"class":"basicInfo-item name", "id":"basic-name"})
            for node in nodes:
                text = node.text.strip()
                if re.match(r'主[\s|\\]+演',text):
                    zhuyan_all.extend(parse_names_with_link(node))
                elif re.match(r"导[\s|\\]+演",text):
                    daoyan_all.extend(parse_names_with_link(node))
                elif text in {"出品时间","首映日期","上映时间","首播时间"}:
                    dates_all.append(parse_date(node))            
        zhuyan_all = sorted(list(set(zhuyan_all)))
        daoyan_all = sorted(list(set(daoyan_all)))
        date_text = min_date(dates_all)
        if zhuyan_all != [] or daoyan_all != []:
            with open(saved_path,"w",encoding='utf-8') as f:
                json.dump({"zhuyan":zhuyan_all,"daoyan":daoyan_all,"date":date_text},f,indent=4,ensure_ascii=False)
        return 

       
def crawler_celebrites_entry():
    with open(common_data['entry'],'r',encoding='utf-8') as f:            
        names = f.read()
    names = [n.strip() for n in names.splitlines()]
    for name in tqdm(names):
        Celebrity(name).get_films()
        
def crawler_films():
    films = []
    for celebrity in os.listdir(celebrities_dir):
        with open(os.path.join(celebrities_dir,celebrity),'r',encoding='utf-8') as f:
            films.extend(json.load(f)['films'])
    films = sorted(list(set(films))) 
    for filmname in tqdm(films):
        try:
            Film(filmname).get_info()
        except Exception as e:
            print(e)
           
def search_daoyan_from_film():
    films = []
    for film in os.listdir(films_dir):
        with open(os.path.join(films_dir,film),'r',encoding='utf-8') as f:
            films.append((film,json.load(f))) 
    daoyan2zhuyan = defaultdict(list)
    daoyan2film = defaultdict(list)
    for film in tqdm(films):
        name, info = film[0],film[1]
        for daoyan in info['daoyan']:
            daoyan2zhuyan[daoyan].extend(info['zhuyan'])
            daoyan2film[daoyan].append(name)
    for daoyan in tqdm(daoyan2film.keys()):
        daoyan_filename = make_valid_filename(daoyan)
        with open(os.path.join(director_dir,daoyan_filename+".json"),'w',encoding='utf-8') as f:
            ctx = {
                "yanyuan":daoyan2zhuyan[daoyan],
                "film":daoyan2film[daoyan]
            }
            json.dump(ctx,f,ensure_ascii=False,indent=4)
    return
    
             
     
            
crawler_celebrites_entry() 
crawler_films()
#search_daoyan_from_film()


#Celebrity("梁朝伟").get_films()
#Film("2008分之一").get_info()