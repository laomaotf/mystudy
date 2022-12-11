- [SC2](#sc2)
  - [游戏信息](#游戏信息)
  - [SC2 APIs](#sc2-apis)
    - [DeepMind的APIs](#deepmind的apis)
    - [安装](#安装)
    - [更加新的APIs](#更加新的apis)
- [GYM](#gym)
  - [安装](#安装-1)
- [baselines3](#baselines3)
  - [安装](#安装-2)


# SC2
## 游戏信息
[人族](https://liquipedia.net/starcraft2/Terran_Units_(Legacy_of_the_Void))  
[神族](https://liquipedia.net/starcraft2/Protoss_Units_(Legacy_of_the_Void))  
[虫族](https://liquipedia.net/starcraft2/Zerg_Units_(Legacy_of_the_Void))

## SC2 APIs
### DeepMind的APIs
[pysc2](https://github.com/deepmind/pysc2)  
[pysc2-doc](https://raw.githubusercontent.com/deepmind/pysc2/master/docs/environment.md)



### 安装  
推荐python3.8+
```bash
pip install pysc2
```

提示和protobuf相关问题，可以降低其版本到3.20.0
```bash
pip install protobuf==3.20.0
```

地图类型有很多种mini-game, melee，各自要放如相应目录才能被APIs访问.
mini-game是经过设计的简单游戏，其中对每一个动作都设置了reward，适合作为一些RL模型的初始研究集合
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

### 更加新的APIs

[BurnySc2](https://github.com/BurnySc2/python-sc2)   
[BurnySc2-doc](https://burnysc2.github.io/python-sc2/docs/text_files/introduction.html)

# GYM
[GYM-APIs](https://github.com/openai/gym)  
[GYM-doc](https://www.gymlibrary.dev/)

## 安装

```bash
pip install gym[all]
```


# baselines3

[baselines3](https://github.com/DLR-RM/stable-baselines3)
[baselines3-doc](https://stable-baselines3.readthedocs.io/en/master/)

## 安装
```bash
pip install stable-baselines3[extra]
```
