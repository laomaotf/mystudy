import requests
from bs4 import BeautifulSoup
import re

import json

with open("common.json",'r') as f:
    common_data = json.load(f)

session = requests.Session()
headers = {
    "User-Agent":common_data['headers']['User-Agent'],
    "Accept": common_data['headers']['Accept'],
    "Accept-Language": common_data['headers']['Accept-Language']}
url = "http://www.user-agent.cn/"
req = session.get(url,headers= headers)
bs = BeautifulSoup(req.text,'html.parser')
items = bs.find(text=re.compile("Mozilla*"))
print(items)
