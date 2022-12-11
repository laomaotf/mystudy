import gym
from gym import spaces
import numpy as np 
import json,copy
import time
import cv2
import warnings
import sys,os
import subprocess
import shutil
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.env_checker import check_env
import random

class CEnv(gym.Env):
    metadata = {"render_modes":["human"],"render_fps":4}
    def __init__(self):
        super().__init__()
        self.action_names = ["up","down","left","right"]
        self.map_size = 64
        # name of observation_space / action_space is determined by check_env()
        # observation_space is input of alg
        # action_space is output of alg
        self.observation_space = spaces.Box(low=-self.map_size,high=self.map_size,shape=(self.map_size*2,),dtype=np.float32)
        self.action_space = spaces.Discrete(len(self.action_names))
        self.minimap, self.position = self._build_minimap(size = self.map_size)
            
    def _build_minimap(self,size = 64):
        minimap = np.zeros((size,size),np.int32)
        return minimap, (size//2,size//2)
    def _refresh_minimap(self, hardlevel = 0.02):
        h,w = self.minimap.shape
        for x in range(w):
            ball_exist = self.minimap[:,x].max()
            if ball_exist > 0:
                ball_in_col = self.minimap[:,x].argmax()
                ball_in_col += 1
                self.minimap[:,x] = 0 #clean ball in this column
                if ball_in_col < h:
                    self.minimap[ball_in_col,x] = 1 #move down
                continue
            if random.uniform(0, 1) > hardlevel:
                continue #skip this column
            self.minimap[0,x] = 1 #add new ball
        return 
    def _get_obs(self, anchor):
        ay,ax = anchor
        _,w = self.minimap.shape
        obs = np.zeros((w*2,),dtype=np.float32)
        for x in range(w):
            ball_in_col = 0
            ball_exist = self.minimap[:,x].max()
            if ball_exist > 0:
                ball_in_col = self.minimap[:,x].argmax()
            obs[x*2] = ball_in_col - ay
            obs[x*2+1] = x - ax
        return obs
            
    def step(self, action):
        action_name = self.action_names[action] 
        y,x = self.position
        #1--apply action
        if action_name == "up":
            y -= 1
        elif action_name == "down":
            y += 1
        elif action_name == "left":
            x -= 1
        elif action_name == "right":
            x += 1
        h,w = self.minimap.shape
        y = min(max([1,y]),h-2)
        x = min(max([1,x]),w-2) 
        self.position = (y,x) 
       
        self._refresh_minimap() 
        self.render()
   
        #2--obs
        obs = self._get_obs(self.position)
        
        #3--reward
        reward = 0.1
        if self.minimap[y,x] != 0: #failed
            return obs, -100., True, {}
        return obs, reward, False, {} 
             
    def reset(self):
        self.minimap, self.position = self._build_minimap(size = self.map_size)
        obs = self._get_obs(self.position)
        return obs
    
    def render(self,mode="human"):
        if mode != "human":
            raise Exception(f"mode must be human")
        h,w = self.minimap.shape
        R = 10
        H,W = h * R, w * R
        vismap = copy.deepcopy(self.minimap)
        m0, m1 = self.minimap.min(), self.minimap.max()
        vismap = np.clip((vismap - m0) * 255 / (m1 - m0),0,255).astype(np.uint8)
        vismap = cv2.resize(vismap,(W,H),interpolation=cv2.INTER_NEAREST)
        vismap = cv2.cvtColor(vismap,cv2.COLOR_GRAY2BGR)
        
        
        y, x = self.position
        Y,X = y * R, x * R
        cv2.circle(vismap,(X,Y),R//2,(255,0,0),3)
        
        cv2.imshow("game",vismap) 
        cv2.waitKey(50)
        return
    def close(self):
        
        cv2.destroyAllWindows()

def is_env_ok():
    env = CEnv()
    env.reset()
    check_env(env)
    
def test_env():
    env = CEnv()
    obs = env.reset()
    for _ in range(100):
        act = env.action_space.sample()
        obs,rew,gg,_ = env.step(act)
        env.render()
        if gg:
            break
        print(act,rew)
    env.close()
    
if __name__ == "__main__":
    is_env_ok()
    test_env()

