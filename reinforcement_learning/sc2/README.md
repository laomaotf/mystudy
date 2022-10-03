

# SC2 游戏信息
[人族](https://liquipedia.net/starcraft2/Terran_Units_(Legacy_of_the_Void))  
[神族](https://liquipedia.net/starcraft2/Protoss_Units_(Legacy_of_the_Void))  
[虫族](https://liquipedia.net/starcraft2/Zerg_Units_(Legacy_of_the_Void))

# SC2 APIs

## DeepMind的APIs
[pysc2](https://github.com/deepmind/pysc2)  
[pysc2-doc](https://raw.githubusercontent.com/deepmind/pysc2/master/docs/environment.md)

## 安装  
推荐python3.8+
```bash
pip install pysc2
```

提示和protobuf相关问题，可以降低其版本到3.20.0
```bash
pip install protobuf==3.20.0
```

地图类型有很多种mini-game, melee，各自要放如相应目录才能被APIs访问
```
STARCRAFT II  
├─Interfaces
├─Maps
│  ├─Melee  
│  │      Empty128.SC2Map  
│  │      Flat128.SC2Map
│  │      Flat32.SC2Map
│  │      Flat48.SC2Map
│  │      Flat64.SC2Map
│  │      Flat96.SC2Map
│  │      Simple128.SC2Map
│  │      Simple64.SC2Map
│  │      Simple96.SC2Map
│  └─mini_games
│          BuildMarines.SC2Map
│          CollectMineralsAndGas.SC2Map
│          CollectMineralShards.SC2Map
│          DefeatRoaches.SC2Map
│          DefeatZerglingsAndBanelings.SC2Map
│          FindAndDefeatZerglings.SC2Map
│          MoveToBeacon.SC2Map
├─SC
├─SC2Data  
├─Support  
└─Versions
```

## 更加新的APIs

[BurnySc2](https://github.com/BurnySc2/python-sc2)   
[BurnySc2-doc](https://burnysc2.github.io/python-sc2/docs/text_files/introduction.html)

