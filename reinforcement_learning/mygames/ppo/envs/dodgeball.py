import gym
from gym import spaces
import numpy as np 
import copy
import cv2
import os
import math
import random
import copy


SAVE_TRAJECTORY = False

USE_IMAGE_OBS = True


class CEnv(gym.Env):
    metadata = {"render_modes":["human"],"render_fps":4}
    def __init__(self,hparam):
        super().__init__()
        self.action_names = ["up","down","left","right"]
        self.map_size = hparam.get("map_size",16)
        self.hparam = hparam
        self.obs_vis = None
        self.obs_ratio = hparam.get('env_obs_resize_ratio',16)
        self.observation_space = spaces.Box(low=hparam.get('env_obs_min',0),high=hparam.get('env_obs_max',255),
                                            shape=(3,self.map_size * self.obs_ratio,self.map_size * self.obs_ratio),
                                            dtype=np.float32)
        self.action_space = spaces.Discrete(len(self.action_names))
        self.minimap, self.position = self._build_minimap(size = self.map_size)
        self.apple_position = self._put_apple()
        self.epoch = 0
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
    def _get_obs(self, anchor):
        apple_y,apple_x = self.apple_position
        ay,ax = anchor
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
        cx, cy = int((apple_x + 0.5) * self.obs_ratio), int((apple_y + 0.5) * self.obs_ratio)
        cv2.circle(obs,(cx,cy), self.obs_ratio//2, (0,1,1),3)
        self.obs_vis = copy.deepcopy(obs) * 255
        return obs.astype(np.float32).transpose(2,0,1)[None,:,:,:] \
            * (self.hparam.get('env_obs_max',255) - self.hparam.get('env_obs_min',0)) + self.hparam.get('env_obs_min',0)
            
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
        x = x % w
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
            reward = 1.0
        #3-update env
        self._refresh_minimap() 
        if self.minimap[y,x] != 0: 
            reward = -500
            gameover = True
        #4--get obs
        obs = self._get_obs(self.position)
        #self.render()
        return obs, reward, gameover, {} 
             
    def reset(self):
        self.epoch += 1
        self.minimap, self.position = self._build_minimap(size = self.map_size)
        self.apple_position = self._put_apple()
        obs = self._get_obs(self.position)
        return obs
        
    def render(self,mode="human"):
        cv2.imshow("game",self.obs_vis)
        cv2.waitKey(5)
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

