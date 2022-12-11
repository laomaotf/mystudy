import pickle
import time

"""
info = {
    "action": string,
    "reward": float,
    "iteration": int,
    "supply_workers": int,
    "supply_cap": int,
    "supply_left": int,
    "source": string,
    "game_over": int,
}
"""

class INFO(object):
    def __init__(self,source_name):
        self.data = {
            "action": "none",
            "reward" : 0,
            "iteration" : 0,
            "supply_workers" : 0,
            "supply_cap" : 0,
            "game_over" : 0,
            "supply_left":0,
            "map_minerals_total":0,
            "map_minerals_left":0,
            "rew_type":""
            
        }
        self.source_name = source_name
                
    def _get_or_set(self,key,value):
        if value is None:
            return self.data[key]
        if key not in self.data.keys():
            raise KeyError(f"{key}")
        self.data[key] = value
        return self.data[key]
    def map_minerals_total(self,value=None):
        return self._get_or_set("map_minerals_total",value)
    def map_minerals_left(self,value=None):
        return self._get_or_set("map_minerals_left",value)
    def rew_type(self,value=None):
        return self._get_or_set("rew_type",value)
    def action(self,value=None):
        return self._get_or_set("action",value)
    def reward(self,value = None):
        return self._get_or_set("reward",value)
    def iteration(self,value=None):
        return self._get_or_set("iteration",value)
    def supply_workers(self,value=None):
        return self._get_or_set("supply_workers",value)
    def supply_cap(self,value=None):
        return self._get_or_set("supply_cap",value)
    def supply_left(self,value=None):
        return self._get_or_set("supply_left",value)
    def game_over(self,value=None):
        return self._get_or_set("game_over",value)
    

    
    def read(self):
        while True:
            try:
                with open("info.dat",'rb') as f:
                    info = pickle.load(f) 
                if info['source'] != self.source_name:
                    break
                time.sleep(0.05)
            except Exception as e:
                time.sleep(0.05)
                pass
        self.data = info
        return

    def write(self):
        self.data['source'] = self.source_name
        with open("info.dat",'wb') as f:
            pickle.dump(self.data,f)
        return