# -*- coding: UTF-8 -*-
from imaplib import Commands
from multiprocessing.connection import wait
#from turtle import position
#from tkinter import UNITS
from sc2 import maps
from sc2.player import Bot, Computer
from sc2.main import run_game
from sc2.data import Race, Difficulty
from sc2.bot_ai import BotAI
from sc2.ids.unit_typeid import UnitTypeId
from sc2.ids.ability_id import AbilityId
import os,sys,cv2
from configs.config import CONFIG
import pickle
import time
import numpy as np
import random
from sc2.unit import Unit
from sc2.units import Units
from configs.config import CONFIG
               
class TerrranAI(BotAI):
    async def build_CommandCenter(self):
        if not self.can_afford(UnitTypeId.COMMANDCENTER):
            return
        worker_candidates = self.workers.filter(lambda worker: (worker.is_collecting or worker.is_idle) and worker.tag not in self.unit_tags_received_action)
        # Worker candidates can be empty
        if worker_candidates:
            map_center = self.game_info.map_center
            position_towards_map_center = self.start_location.towards(map_center, distance=5)
            placement_position = await self.find_placement(UnitTypeId.COMMANDCENTER, near=position_towards_map_center, placement_step=1) 
            # Placement_position can be None
            if placement_position:
                build_worker = worker_candidates.closest_to(placement_position)
                build_worker.build(UnitTypeId.COMMANDCENTER, placement_position)
        return
    def get_one_commandCenter_in_free(self):
        ccs = self.townhalls.filter(lambda cc : cc.is_idle)
        if not ccs:
            return None
        return  ccs.random
    
    def get_one_barrack(self):
        barracks_all = self.structures.filter(lambda x: x.type_id == UnitTypeId.BARRACKS and int(x.build_progress) == 1 )
        if barracks_all == []:
            return None
        return barracks_all.random
    
    

    async def build_SCV(self,commandCenter):
        if self.can_afford(UnitTypeId.SCV):
            commandCenter.train(UnitTypeId.SCV)
        return            
     
    async def build_Supplydepot(self, step_max = 5, step_min = 1):
        if self.can_afford(UnitTypeId.SUPPLYDEPOT) and self.already_pending(UnitTypeId.SUPPLYDEPOT) == 0:
            step = random.uniform(0.1,1.0) * (step_max - step_min) + step_min
            pos = self.start_location + step * self.start_location.direction_vector(self.enemy_start_locations[0])
            await self.build(UnitTypeId.SUPPLYDEPOT,near=pos)
            #print('build supplyDepot')
        return
            
    async def build_Refinery(self,commandCenter):
        if self.can_afford(UnitTypeId.REFINERY) and 0 == self.already_pending(UnitTypeId.REFINERY): 
            for target_vespene_geyser in self.vespene_geyser.closer_than(20,commandCenter):        
                await self.build(UnitTypeId.REFINERY,near=target_vespene_geyser)
                break
        return
    async def build_Barracks(self, step_max=10, step_min = 5, N_max = 1000):
        if self.can_afford(UnitTypeId.BARRACKS) and self.units_created[UnitTypeId.BARRACKS] + self.already_pending(UnitTypeId.BARRACKS) < N_max:
            step = random.uniform(0.1,1.0) * (step_max  - step_min) + step_min
            pos = self.start_location + step * self.start_location.direction_vector(self.enemy_start_locations[0])
            await self.build(UnitTypeId.BARRACKS,near=pos)
        return 
    
    async def build_Marine(self):
        if self.can_afford(UnitTypeId.MARINE):
            barrack = self.get_one_barrack()
            if barrack:
                ret = barrack.train(UnitTypeId.MARINE) 
        return
   
    async def attack_by(self,army, target,group_size = 1):
        if isinstance(army,UnitTypeId):
            if army == UnitTypeId.MARINE and (self.units(UnitTypeId.MARINE).amount > group_size):
                for marine in self.units(UnitTypeId.MARINE):
                    marine.attack(target)
        elif isinstance(army,Unit):
            army.attack(target)
        return
    
    async def new_CommandCenter(self):
        if self.can_afford(UnitTypeId.COMMANDCENTER):
            await self.expand_now()
            
                 
    def get_enemies_in_range(self,army,distance=10):
        if army is not None:
            enemies = self.enemy_units.closer_than(distance,army)
            if enemies.amount > 0:
                return enemies
            enemies = self.enemy_structures.closer_than(distance,army)
            if enemies.amount > 0:
                return enemies
        enemies = self.enemy_units
        if enemies.amount > 0:
            return enemies
        return self.enemy_structures
        
    async def patrol_by(self,army,target, distance=10):
        if isinstance(army, UnitTypeId):
            for one in self.units(army):
                if one.is_idle and one.distance_to(target) < distance:
                    one.patrol(target)
        
                              
        
    async def on_step(self, iteration: int):
        while True:
            try:
                with open("info.dat",'rb') as f:
                    info = pickle.load(f) 
                if info['action'] is not None:
                    break
                time.sleep(0.05)
            except Exception as e:
                time.sleep(0.05)
                pass
        action = info['action']
        cfg = CONFIG(info['config_file'])
        await self.distribute_workers(1.0)
        
        await self.patrol_by(UnitTypeId.MARINE, self.start_location)
       
        if action == "build_SCV":
            #print(f"agent run action {action}: Build SCV")
            cc = self.get_one_commandCenter_in_free()
            if cc is not None:
                await self.build_SCV(cc) 
        elif action == "build_Supplydepot":
            #print(f"agent run action {action}: Build SupplyDep")
            await self.build_Supplydepot()
        elif action == "build_Refinery":
            #print(f"agent run action {action}: Build Refinery")
            cc = self.get_one_commandCenter_in_free()
            if cc is not None:
                await self.build_Refinery(cc)
        elif action == "build_Barracks":
            #print(f"agent run action {action}: Build Barracks")
            await self.build_Barracks()
        elif action == "build_Marine":
            #print(f"agent run action {action}: Build Marine")
            await self.build_Marine()
        elif action == "attack_by_Marine":
            #print(f"agent run action {action}: Attack by Marines")
            for marine in self.units(UnitTypeId.MARINE):
                if marine.is_attacking:
                    continue
                enemies = self.get_enemies_in_range(marine)
                if enemies.amount > 0:
                    await self.attack_by(marine, enemies.random) 
        elif action == "attack_by_MarineGroup":
            enemies = self.get_enemies_in_range(None)
            if enemies.amount > 0:
                await self.attack_by(UnitTypeId.MARINE, enemies.random, group_size=10)             
        elif action == "build_commandCenter":
            await self.new_CommandCenter()
        elif action == "patrol_by_marine":
            if iteration - info['last_patrol'] > 200:
                if self.units(UnitTypeId.MARINE).amount > 0:
                    marine = self.units(UnitTypeId.MARINE).random 
                    #marine = self.units(UnitTypeId.MARINE).first
                    targets = self.mineral_field.filter(lambda x: not x.is_visible)
                    if targets.amount == 0:
                        targets = self.mineral_field
                    target = targets.further_than(10,marine).random.position
                    marine.patrol(target)
                    info['last_patrol'] = iteration
        else:
            raise Exception(f"unknown action {action}")
       
        info['obs'] = info['obs'] * 0
        map_w, map_h = self.game_info.map_size.width, self.game_info.map_size.height
        #print(f"{map_w}x{map_h}")
        #print(f"{self.start_location}")
        win_h, win_w = info['obs'].shape[0:2]
        for res in self.mineral_field: 
            x,y = res.position.x,res.position.y
            x,y = int(x * win_w / map_w), int(y * win_h / map_h)
            color = cfg.get('observation.color.mineral_visible')
            lighting = np.clip(res.mineral_contents / 2000,0,1)
            if res.is_visible:
                info['obs'][y,x] = [int(c*lighting) for c in color]
            else:
                info['obs'][y,x] = cfg.get('observation.color.mineral_invisible')
                
        for res in self.vespene_geyser: 
            x,y = res.position.x,res.position.y
            x,y = int(x * win_w / map_w), int(y * win_h / map_h)
            color = cfg.get('observation.color.vespene_geyser_visible')
            lighting = np.clip(res.mineral_contents / 2000,0,1)
            if res.is_visible:
                info['obs'][y,x] = [int(c*lighting) for c in color]
            else:
                info['obs'][y,x] = cfg.get('observation.color.vespene_geyser_invisible')
                
        for res in self.structures: 
            x,y = res.position.x,res.position.y
            x,y = int(x * win_w / map_w), int(y * win_h / map_h)
            color = cfg.get('observation.color.structures')
            lighting = np.clip(res.health_percentage,0,1)
            info['obs'][y,x] = [int(c*lighting) for c in color]
                
        for res in self.units: 
            x,y = res.position.x,res.position.y
            x,y = int(x * win_w / map_w), int(y * win_h / map_h)
            if res.type_id == UnitTypeId.MARINE:
                color = cfg.get('observation.color.units_marine')
            else:
                color = cfg.get('observation.color.units')
            lighting = np.clip(res.health_percentage,0,1)
            info['obs'][y,x] = [int(c*lighting) for c in color]
        
        
        # for res in self.all_units: #show enemy detected but not visible
        #     if not res.is_enemy:
        #         continue 
        #     if res.is_visible:
        #         continue
        #     x,y = res.position.x,res.position.y
        #     x,y = int(x * win_w / map_w), int(y * win_h / map_h)
        #     info['obs'][y,x] = [127,127,127]
        
        for res in self.enemy_units: 
            x,y = res.position.x,res.position.y
            x,y = int(x * win_w / map_w), int(y * win_h / map_h)
            color = cfg.get('observation.color.enemy_units')
            lighting = np.clip(res.health_percentage,0,1)
            info['obs'][y,x] = [int(c*lighting) for c in color]
            
        for res in self.enemy_structures: 
            x,y = res.position.x,res.position.y
            x,y = int(x * win_w / map_w), int(y * win_h / map_h)
            color = cfg.get('observation.color.enemy_structures')
            lighting = np.clip(res.health_percentage,0,1)
            if not res.is_visible:
                color = cfg.get('observation.color.enemy_structures_invisible')
                lighting = 1.0
            info['obs'][y,x] = [int(c*lighting) for c in color]
            
        if info['flag_debug'] and (info['obs'] is not None):
            vis = cv2.flip(info['obs'],0)
            vis = cv2.resize(vis,(512,512),0,0,interpolation=cv2.INTER_LINEAR)
            cv2.imshow("obs",vis)
            cv2.waitKey(1)
            
        #feedback reward
        reward = cfg('reward.time')
        try:
            for unit in self.units(UnitTypeId.MARINE):
                if unit.is_attacking:
                    for enemy in self.enemy_structures.closer_than(10,unit):
                        if unit.target_in_range(enemy):
                            reward += 1
                    for enemy in self.enemy_units.closer_than(10,unit):
                        if unit.target_in_range(enemy):
                            reward += 1
        except Exception as e:
            print(f"reward exception: {e}")
            reward = 0
         
        #print(f"reward = {reward}")
        info['reward'] = reward
        info['action'] = None
        info['terminated'] = False
        with open('info.dat','wb') as f:
            pickle.dump(info,f)
        return 



def main(cfg):
    RACE_DICT = {
        "Zerg" : Race.Zerg,
        "Terran" : Race.Terran,
        "Protoss" : Race.Protoss 
    }
    
    DIFFICULTY_DICT = {
        "Easy" : Difficulty.Easy,
        "Hard" : Difficulty.Hard,
        "Medium" : Difficulty.Medium 
    }
        
    results = str(run_game(maps.get(cfg.get('game.map')), 
                           [Bot(Race.Terran, TerrranAI()),
                            Computer(RACE_DICT[cfg.get('game.computer.race')], DIFFICULTY_DICT[cfg.get('game.computer.difficulty')])], 
                           realtime=cfg.get("game.realtime")))

    reward = cfg.get('reward.result_victory')
    if results == "Result.Defeat":
        reward = cfg.get('reward.result_defeat')

    try:
        with open("info.dat",'rb') as f:
            info = pickle.load(f) 
    except Exception as e:
        pass
    info['action'] = None
    info['reward'] = reward
    info['obs'] = info['obs'] * 0
    info['terminated'] = True
    with open("info.dat",'wb') as f:
        pickle.dump(info,f)
    
    with open("results.txt",'a') as f:
        f.write(f"{str(results)}\n")
        
if __name__ == "__main__":
    cfg = CONFIG(sys.argv[1])
    main(cfg)