# -*- coding: UTF-8 -*-
from sc2 import maps
from sc2.player import Bot, Computer
from sc2.main import run_game
from sc2.data import Race, Difficulty
from sc2.bot_ai import BotAI
from sc2.ids.unit_typeid import UnitTypeId
import logging
import os,time
from info import INFO
import pandas as pd
from matplotlib import pyplot as plt
plt.ion()


logging.basicConfig(level=logging.INFO)

class TerrranAI(BotAI):
    def __init__(self):
        super().__init__()
        self.map_minerals_total = -1
        self.history = {
            "reward":[0],
            "minerals":[50],
        }
        
    async def build_SCV(self):
        if self.can_afford(UnitTypeId.SCV):
            self.train(UnitTypeId.SCV) #???????????? await?
            #logging.info(f"create one SCV")
        return            
     
    async def build_Supplydepot(self):
        if self.can_afford(UnitTypeId.SUPPLYDEPOT):
            await self.build(UnitTypeId.SUPPLYDEPOT,near= self.townhalls.random)
            #logging.info(f"create one Supplydepot")
        return
    
                
    def print_game_info(self,iteration):
        if iteration % 5 != 0:
            return
        logging.info(f"iter: {iteration}")
        logging.info(f"\t 矿物: {self.minerals}")
        
        logging.info(f"\t 天然气: {self.vespene}")
        logging.info(f"\t 军队人数(有效): {self.supply_army}")
        logging.info(f"\t 后勤人数(有效): {self.supply_workers}")
        logging.info(f"\t 人数上限: {self.supply_cap}")
        logging.info(f"\t 人数空额: {self.supply_left}")
         
        logging.info(f"\t 空闲后勤/后勤人数: {self.idle_worker_count}/{len(self.workers)}")
        logging.info(f"\t 基地: {len(self.townhalls)}")
        logging.info(f"\t 气矿: {len(self.gas_buildings)}")
        
        return 
    
    def draw(self):
        Y = self.history['reward']
        X = [x for x in range(len(Y))]
        plt.plot(X,Y)
        plt.draw()
        plt.pause(0.01)
        return
    async def on_step(self, iteration: int):

        #self.print_game_info(iteration)
        #return 
         
        info = INFO(__file__)
        info.read()
        action = info.action()
        #self.print_game_info(iteration)
        await self.distribute_workers(1.0)
        #print(action)
        if action == "build_SCV":
           await self.build_SCV()
        elif action == "build_Supplydepot":
           await self.build_Supplydepot()
        else:
           pass  
        rew_type = info.rew_type()
        if rew_type == "leftMax":
            minerals = self.minerals
            reward = float(minerals) / (iteration + 1)
        elif rew_type == "gatherMax":
            map_minerals_left = 0
            for mfd in self.mineral_field:
                map_minerals_left += mfd.mineral_contents 
            if self.map_minerals_total < 1:
                self.map_minerals_total = map_minerals_left
                self.history['minerals'][0] = map_minerals_left
            minerals = map_minerals_left
            #reward = float(self.map_minerals_total - map_minerals_left) / (iteration + 1)
            reward = self.history['minerals'][-1] - minerals
            #reward_delta = reward - self.history['reward'][-1]
            #print(iteration, self.map_minerals_total, map_minerals_left, reward)
            
        self.history['minerals'].append( minerals )
        self.history['reward'].append(reward)
        
        #if 0 == (iteration % 10):
        #    df = pd.DataFrame({"reward":self.history['reward'], "minerals":self.history['minerals']})
        #    df.to_csv(os.path.join('output',"log.csv"),sep=',',index=True,header={"reward","minerals"})
        
        #print(self.history['reward'])
        info.map_minerals_left(minerals)
        info.map_minerals_total(self.map_minerals_total)
        info.iteration(iteration)
        info.reward(reward)
        info.supply_workers(self.supply_workers)
        info.supply_cap(self.supply_cap)
        info.supply_left(self.supply_left)
        info.game_over(0)
        info.write()

def main():
    run_game(maps.get("CollectMineralsAndGas"), [Bot(Race.Terran, TerrranAI()), Computer(Race.Terran,Difficulty.Hard)], realtime=False)
    info = INFO(__file__) 
    info.game_over(1)
    info.reward(0)
    info.write() 
        
if __name__ == "__main__":
    main()
                         