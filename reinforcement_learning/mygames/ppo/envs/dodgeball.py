import gym
from gym import spaces
import numpy as np 
import json,copy
import time
import cv2
import warnings
import sys,os
import math
import subprocess
import shutil
import random
from datetime import datetime
import pickle
import copy


SAVE_TRAJECTORY = False

USE_IMAGE_OBS = True


class CEnv(gym.Env):
    metadata = {"render_modes":["human"],"render_fps":4}
    def __init__(self,hparam):
        super().__init__()
        #random.seed(datetime.now().timestamp())
        self.action_names = ["up","down","left","right"]
        self.map_size = hparam.get("map_size",16)
        self.hparam = hparam
        # name of observation_space / action_space is determined by check_env()
        # observation_space is input of alg
        # action_space is output of alg
        if USE_IMAGE_OBS:
            self.obs_vis = None
            self.obs_ratio = hparam.get('env_obs_resize_ratio',16)
            self.observation_space = spaces.Box(low=hparam.get('env_obs_min',0),high=hparam.get('env_obs_max',255),
                                                shape=(3,self.map_size * self.obs_ratio,self.map_size * self.obs_ratio),
                                                dtype=np.float32)
        else:
            self.observation_space = spaces.Box(low=-self.map_size,high=self.map_size,shape=(5+2,),dtype=np.float32)
        self.action_space = spaces.Discrete(len(self.action_names))
        self.minimap, self.position = self._build_minimap(size = self.map_size)
        self.apple_position = self._put_apple()
        self.epoch = 0
        self.trajectory = {
            "size":self.minimap.shape,
            "apple": [],
            "player": [],
            "bomb": []
        }
    def _put_apple(self):
        return random.randint(self.map_size//2,self.map_size-3), random.randint(3, self.map_size-3)
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
    def _calc_dist(self,p0, p1,size):
        h,w = size
        y0,x0 = p0
        y1,x1 = p1
        d0 = (x0-x1)**2 + (y0-y1)**2
        
        xmin, xmax = min([x0,x1]), max([x0,x1])
        x0 = xmin + w
        x1 = xmax
        d1 = (x0-x1)**2 + (y0-y1)**2
        return math.sqrt(min([d0,d1]))
    def _get_obs(self, anchor):
        
        apple_y,apple_x = self.apple_position
        ay,ax = anchor
        if USE_IMAGE_OBS:
            obs = self.minimap[:,:,None]
            obs = np.concatenate([obs,obs,obs],axis=-1).astype(np.uint8)
            obs = cv2.resize(obs,(self.map_size*self.obs_ratio, self.map_size*self.obs_ratio),0,0,cv2.INTER_NEAREST)
            if self.hparam.get('env_obs_custom_draw',0):
                obs = np.zeros_like(obs)
                for y in range(self.minimap.shape[0]):
                    for x in range(self.minimap.shape[1]):
                        val = int(self.minimap[y,x])
                        if val < 1:
                            continue
                        obs[y*self.obs_ratio:(y+1)*self.obs_ratio, x*self.obs_ratio:(x+1)*self.obs_ratio,:] = 1 
                            
            cx, cy = int((ax + 0.5) * self.obs_ratio), int((ay + 0.5) * self.obs_ratio)
            cv2.circle(obs,(cx,cy), self.obs_ratio//2, (0,1,0),3)
            #print(ax,ay, cx,cy)
            cx, cy = int((apple_x + 0.5) * self.obs_ratio), int((apple_y + 0.5) * self.obs_ratio)
            cv2.circle(obs,(cx,cy), self.obs_ratio//2, (0,1,1),3)
            self.obs_vis = copy.deepcopy(obs)
            #cv2.imshow("obs",obs * 255)
            #cv2.waitKey(-1)
            return obs.astype(np.float32).transpose(2,0,1)[None,:,:,:] \
                * (self.hparam.get('env_obs_max',255) - self.hparam.get('env_obs_min',0)) + self.hparam.get('env_obs_min',0)
        _,w = self.minimap.shape
        obs = np.zeros((5+2,),dtype=np.float32)
        for rx in range(ax-2,ax+3,1):
            x = (rx + w) % w
            ball_in_col = 0
            ball_exist = self.minimap[:,x].max()
            if ball_exist > 0:
                ball_in_col = self.minimap[:,x].argmax()
            #obs[rx-ax+2] = self._calc_dist(anchor,(ball_in_col,rx),self.minimap.shape) + (ax==x)
            obs[rx-ax+2] = ball_in_col - ay + (ax==x)
        if apple_y >= 0 and apple_x >= 0:
            #obs[-1] = self._calc_dist(self.apple_position, anchor,self.minimap.shape)
            obs[-2] = apple_x - ax
            obs[-1] = apple_y - ay
        return obs
    def _calc_apple_adv(self,position):
        points = [[0,0],[0,self.map_size-1],[self.map_size-1, self.map_size-1],[self.map_size-1,0],self.apple_position]
        dist = list(map(lambda p: self._calc_dist(p,position,self.minimap.shape),points))
        dist = np.array(dist)
        rev_dist = dist[-1] / dist[0:-1].max()
        #reward *= (1 - rev_dist)
        return max([0.25 - rev_dist,0])
        
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
        #x = min(max([1,x]),w-2) 
        x = x % w
        #y = h - 1 ####
        #x = self.apple_position[1]
        self.position = (y,x) 
       
        #2--reward
        apple_y, apple_x = self.apple_position
        reward = 0.0
        gameover = False
        if y == apple_y and x == apple_x:
            reward = 100
            self.apple_position = self._put_apple()
        elif self.minimap[y,x] != 0 : #failed
            reward = -500
            gameover = True
        else:
            reward = 1.0 +  self._calc_apple_adv(self.position) * 50 #数量级的差异
        #3-update env
        self._refresh_minimap() 
        if self.minimap[y,x] != 0: 
            reward = -500
            gameover = True
        #4--get obs
        obs = self._get_obs(self.position)
        self.render()
        return obs, reward, gameover, {} 
             
    def reset(self):
        self.epoch += 1
        if SAVE_TRAJECTORY and self.epoch > 1:
            os.makedirs("trajectories",exist_ok=True)
        #random.seed(datetime.now().timestamp())
        self.minimap, self.position = self._build_minimap(size = self.map_size)
        #H,W = self.minimap.shape[0:2]
        #H,W = int(H * self.obs_ratio), int(W * self.obs_ratio)
        self.apple_position = self._put_apple()
        obs = self._get_obs(self.position)
        self.trajectory = {
            "size":self.minimap.shape,
            "apple": [],
            "player": [],
            "bomb": []
        }
        return obs
    def expand_trajectory(self):
        self.trajectory["size"] = self.minimap.shape
        self.trajectory['apple'].append(self.apple_position)
        self.trajectory['player'].append(self.position)
        self.trajectory['bomb'].append(copy.deepcopy(self.minimap))
        return
        
    def render(self,mode="human"):

        if USE_IMAGE_OBS and self.obs_vis is None:
            raise Exception(f"no obs")
        if USE_IMAGE_OBS:
            cv2.imshow("game",self.obs_vis*255)
            cv2.waitKey(5)
            return 
            
        if SAVE_TRAJECTORY:
            self.expand_trajectory()
        
        h,w = self.minimap.shape
        R = self.obs_ratio
        H,W = h * R, w * R
        vismap = copy.deepcopy(self.minimap)
        m0, m1 = self.minimap.min(), self.minimap.max()
        vismap = np.clip((vismap - m0) * 255 / (m1 - m0 + 0.00001),0,255).astype(np.uint8)
        vismap = cv2.resize(vismap,(W,H),interpolation=cv2.INTER_NEAREST)
        vismap = cv2.cvtColor(vismap,cv2.COLOR_GRAY2BGR)
        
        
        y, x = self.position
        Y,X = int((y+0.5) * R), int((x+0.5) * R)
        cv2.circle(vismap,(X,Y),R//2,(0,255,0),1)
        
        y,x = self.apple_position
        #vismap_apple = np.zeros_like(self.minimap,dtype=np.float32)
        if x >= 0 and y >= 0:
            Y,X = y  * R, x * R
            cv2.rectangle(vismap,(X, Y),(X+R, Y+R),(0,255,255),1) 
            
        #     for y in range(h):
        #         for x in range(w):
        #            v = self._calc_apple_adv((y,x))
        #            vismap_apple[y,x] = v
        #     m0, m1 = vismap_apple.min(), vismap_apple.max()
        #     vismap_apple = np.clip((vismap_apple - m0) * 255 / (m1 - m0),0,255).astype(np.uint8)   
        # vismap_apple = cv2.resize(vismap_apple,(W,H),interpolation=cv2.INTER_NEAREST)
         
        #cv2.imshow("game-apple-adv",vismap_apple) 
        cv2.imshow("game",vismap) 
        cv2.waitKey(10)
        return
    def close(self):
        
        cv2.destroyAllWindows()


import yaml

class HARGS_T:
    def __init__(self, path) -> None:
        with open(path,'r') as f:
            self.data = yaml.load(f,Loader=yaml.FullLoader)
        return
    def get(self,key,default_value=None):
        if key in self.data.keys():
            return self.data[key]
        print("not set hyper-param : {}".format(key))
        return default_value
    
def test_env():
    hpt = HARGS_T("../hargs/0001c3d1b.yaml")
    env = CEnv(hpt)
    obs = env.reset()
    for _ in range(10000):
        act = env.action_space.sample()
        obs,rew,gg,_ = env.step(act)
        env.render()
        if gg:
            print("gg")
            break
        print(act,rew)
    env.close()
    
if __name__ == "__main__":
    test_env()

