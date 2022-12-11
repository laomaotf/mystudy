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
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.env_checker import check_env


class CEnv(gym.Env):
    metadata = {"render_modes":["human"],"render_fps":4}
    def __init__(self):
        super().__init__()
        self.action_names = ["up","down","left","right"]
        # name of observation_space / action_space is determined by check_env()
        # observation_space is input of alg
        # action_space is output of alg
        self.observation_space = spaces.Box(low=-1,high=1000,shape=(9,),dtype=np.float32)
        self.action_space = spaces.Discrete(len(self.action_names))
        self.minimap, self.position = self._build_minimap()
        self.trace = [self.position]
            
    def _build_minimap(self,size = None):
        if size is None:
            size = (64,64)
        size_with_wall = (size[0] + 1, size[1] + 1)
        minimap = np.zeros(size_with_wall,np.int32) - 10
        y0,x0 = 1,1
        y1,x1  = size[0] - 5, size[1] - 10
        rew = 1
        for y in range(y0, y1+1):
            minimap[y,x0] = rew
            rew += 1
        for x in range(x0, x1+1):
            minimap[y1,x] = rew
            rew += 1 
        minimap[y1,x1] = rew + 100
        return minimap, (y0,x0)
    def step(self, action):
        action_name = self.action_names[action] 
        y,x = self.position
        lastpos = (y,x)
        
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
        self.trace.append(self.position)
        
        self.render()
   
        #2--obs
        obs = self.minimap[y-1,x-1:x+2].tolist()  + self.minimap[y,x-1:x+2].tolist() + self.minimap[y+1,x-1:x+2].tolist()
        obs = np.array(obs).reshape(-1).astype(np.float32)
        if self.minimap[y,x] == self.minimap.max():
            return obs, 1000.0, True, {}
        #3--reward 
        #reward = self.minimap[y,x] - self.minimap[lastpos[0],lastpos[1]]
        reward = self.minimap[y,x]
        #reward += 0.01
        return obs, reward*1.0, False, {} 
             
    def reset(self):
        self.minimap, self.position = self._build_minimap()
        y,x = self.position
        self.trace = [self.position]
        obs = self.minimap[y-1,x-1:x+2].tolist()  + self.minimap[y,x-1:x+2].tolist() + self.minimap[y+1,x-1:x+2].tolist()
        return np.array(obs).reshape(-1).astype(np.float32)
    
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
       
        
        if len(self.trace) >= 2: 
            for f,t in zip(self.trace[0:-1], self.trace[1:]):
                cv2.line(vismap,(f[1]*R,f[0]*R),(t[1]*R,t[0]*R),(0,255,0),1)
            if len(self.trace) % 100 == 0:
                cv2.imwrite("game.png",vismap)
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
    for _ in range(10):
        act = env.action_space.sample()
        obs,rew,gg,_ = env.step(act)
        env.render()
        if gg:
            break
        print(act,rew)
    env.close()
    
if __name__ == "__main__":
    test_env()

