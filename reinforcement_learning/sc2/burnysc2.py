# -*- coding: UTF-8 -*-
from cgi import print_arguments
from multiprocessing.connection import wait
from sc2 import maps
from sc2.player import Bot, Computer
from sc2.main import run_game
from sc2.data import Race, Difficulty
from sc2.bot_ai import BotAI
from sc2.ids.unit_typeid import UnitTypeId
from sc2.ids.ability_id import AbilityId
import logging
import os

TAG = os.path.basename(__file__)


logging.basicConfig(level=logging.INFO)

class WorkerRushBot(BotAI):
    async def on_step(self, iteration: int):
        if iteration == 0:
            for worker in self.workers:
                worker.attack(self.enemy_start_locations[0])
                
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
    
    def get_one_barrack(self,commandCenter):
        if not self.can_afford(UnitTypeId.BARRACKS):
            return None
        barracks_all = self.structures.filter(lambda x: x.type_id == UnitTypeId.BARRACKS and int(x.build_progress) == 1 )
        if barracks_all == []:
            return None
        return barracks_all.random
    
    

    async def build_SCV(self,commandCenter):
        if self.can_afford(UnitTypeId.SCV):
            commandCenter.train(UnitTypeId.SCV)
            logging.Logger(TAG).info(f"create one SCV")
        return            
     
    async def build_Supplydepot(self,commandCenter):
        if self.can_afford(UnitTypeId.SUPPLYDEPOT) and self.already_pending(UnitTypeId.SUPPLYDEPOT) == 0:
            await self.build(UnitTypeId.SUPPLYDEPOT,near=commandCenter)
            logging.getLogger(TAG).info(f"create one Supplydepot")
        return
            
    async def build_Refinery(self,commandCenter):
        if self.can_afford(UnitTypeId.REFINERY) and 0 == self.already_pending(UnitTypeId.REFINERY): 
            for target_vespene_geyser in self.vespene_geyser.closer_than(20,commandCenter):        
                await self.build(UnitTypeId.REFINERY,near=target_vespene_geyser)
                logging.getLogger(TAG).info(f"create one Refinery")
                break
        return
    async def build_Barracks(self,commandCenter):
        if self.can_afford(UnitTypeId.BARRACKS):
            await self.build(UnitTypeId.BARRACKS,near=commandCenter)
            logging.getLogger(TAG).info("create one Barracks")
        return 
    
    async def build_Marine(self,commandCenter):
        if self.can_afford(UnitTypeId.MARINE):
            barrack = self.get_one_barrack(commandCenter=commandCenter)
            if barrack:
                ret = barrack.train(UnitTypeId.MARINE) 
                logging.getLogger(TAG).info(f"create one Marine {ret}")
        return
   
    async def attack_by(self,unitId, target):
        if unitId == UnitTypeId.MARINE:
            if self.units_created[UnitTypeId.MARINE] % 10 == 0:
                for marine in self.units(UnitTypeId.MARINE):
                    marine.attack(target)
        return
                 
                
    def print_game_info(self,iteration):
        if iteration % 5 != 0:
            return
        logging.getLogger(TAG).info(f"iter: {iteration}")
        logging.getLogger(TAG).info(f"\t 矿物: {self.minerals}")
        logging.getLogger(TAG).info(f"\t 天然气: {self.vespene}")
        logging.getLogger(TAG).info(f"\t 军队人数(有效): {self.supply_army}")
        logging.getLogger(TAG).info(f"\t 后勤人数(有效): {self.supply_workers}")
        logging.getLogger(TAG).info(f"\t 人数上限: {self.supply_cap}")
        logging.getLogger(TAG).info(f"\t 人数空额: {self.supply_left}")
         
        logging.getLogger(TAG).info(f"\t 空闲后勤/后勤人数: {self.idle_worker_count}/{len(self.workers)}")
        logging.getLogger(TAG).info(f"\t 基地: {len(self.townhalls)}")
        logging.getLogger(TAG).info(f"\t 气矿: {len(self.gas_buildings)}")
         
        
        return 
        
        
        
    async def on_step(self, iteration: int):
        self.print_game_info(iteration)
        if self.townhalls:
            await self.distribute_workers(2)
            commandCenter = self.get_one_commandCenter_in_free()
            if commandCenter is None:
                return
            if  len(self.workers) < 16:
                await self.build_SCV(commandCenter=commandCenter)
            if self.supply_left < 5:
                await self.build_Supplydepot(commandCenter=commandCenter)
            if len(self.workers) > 16:
                await self.build_Refinery(commandCenter=commandCenter)
            if len(self.structures.filter(lambda x: x.type_id == UnitTypeId.BARRACKS)) < 2:
                await self.build_Barracks(commandCenter=commandCenter)
            if len(self.structures.filter(lambda x: x.type_id == UnitTypeId.BARRACKS)) > 0:
                await self.build_Marine(commandCenter)
            await self.attack_by(UnitTypeId.MARINE,self.enemy_start_locations[0]) 
        else:
            await self.build_CommandCenter()
                  
run_game(maps.get("Simple64"), [Bot(Race.Terran, TerrranAI()),Computer(Race.Zerg, Difficulty.Hard)], realtime=False)