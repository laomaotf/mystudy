import gym
from gym import spaces
import numpy as np 
import copy
import cv2
import os
import math
import random
import copy


class CEnv(gym.Env):
    metadata = {"render_modes":["human"],"render_fps":4}
    def __init__(self,hparam):
        super().__init__()
        self.action_names = ["up","down","left","right"]
        self.map_size = hparam.get("map_size",256)
        self.apple_radius = 5
        self.bomb_radius = 9
        self.agent_radius = 3
        self.hparam = hparam
        self.observation_space = spaces.Box(low=hparam.get('env_obs_min',0),high=hparam.get('env_obs_max',255),
                                            shape=(3,self.map_size,self.map_size),
                                            dtype=np.float32)
        self.action_space = spaces.Discrete(len(self.action_names))
        self._reset_agent()
        self._reset_apple()
        self.epoch = 0
    def _reset_bombs(self):
        self.bombs = []
        return
    def _reset_apple(self):
        self.apple = np.random.randint(self.map_size//4, self.map_size*3//4, size=2)
        return
    def _reset_agent(self):
        self.agent = np.random.randint(self.agent_radius*2, self.map_size-self.agent_radius*2, size=2)
        return
    def _generate_map(self,color=False):
        data = np.zeros((self.map_size, self.map_size),np.uint8)
        if color:
            data = cv2.cvtColor(data, cv2.COLOR_GRAY2BGR)
        return data
    def _refresh_minimap(self, hardlevel = 0.002):
        step = random.randint(self.bomb_radius//2, self.bomb_radius * 5)
        bomb_positions = self._generate_map()
        for (y,x) in self.bombs:
            bomb_positions[y,x] = 1
        speed = self.bomb_radius//4
        for x in range(step,self.map_size,step):
            ball_exist = bomb_positions[:,x].max()
            if ball_exist > 0:
                ball_in_col = bomb_positions[:,x].argmax()
                ball_in_col += speed
                bomb_positions[:,x] = 0 #clean ball in this column
                if ball_in_col < self.map_size:
                    bomb_positions[ball_in_col,x] = 1 #move down
                continue
            if random.uniform(0, 1) > hardlevel:
                continue #skip this column
            bomb_positions[0,x] = 1 #add new ball
        Y,X = np.where(bomb_positions)
        self.bombs = []
        for y,x in zip(Y,X):
            self.bombs.append((y,x))
        return 
    def _get_obs(self):
        obs = self._generate_map(color=True)
        for (y,x) in self.bombs:
            cv2.circle(obs,(x,y),self.bomb_radius,(0,0,1),-1)
        cv2.circle(obs,self.agent[::-1],self.agent_radius,(0,1,1),-1)
        cv2.circle(obs,self.apple[::-1],self.apple_radius,(1,0,0),-1)
        return obs.astype(np.float32).transpose(2,0,1)[None,:,:,:] \
            * (self.hparam.get('env_obs_max',255) - self.hparam.get('env_obs_min',0)) + self.hparam.get('env_obs_min',0)
            
    def _judge_game(self):
        reward = 0.0
        gameover = False
        objs = np.array(self.bombs + [self.apple]).reshape((-1,2))
        agent = np.array(self.agent).reshape((-1,2))
        dist = np.sqrt(np.sum((agent - objs) ** 2,axis=-1))
        if dist[-1] < self.apple_radius + self.agent_radius:
            reward = 100.0
            self._reset_apple()
        elif len(self.bombs) > 0:
            if np.any(dist[0:-1] < self.agent_radius + self.bomb_radius):
                reward = -500.0
                gameover = True
            else:
                reward = 1.0
        return reward, gameover
    def step(self, action):
        action_name = self.action_names[action]        
        y,x = self.agent
        speed = self.agent_radius
        #1--apply action
        if action_name == "up":
            y -= speed
        elif action_name == "down":
            y += speed
        elif action_name == "left":
            x -= speed
        elif action_name == "right":
            x += speed
        y = min(max([self.agent_radius*2,self.map_size]),self.map_size-2*self.agent_radius)
        x = x % self.map_size
        self.agent = (y,x) 
       
        #2--reward

        
        reward,gameover = self._judge_game()
        #3-update env
        self._refresh_minimap() 
        reward,gameover = self._judge_game()
        #4--get obs
        obs = self._get_obs()
        #self.render()
        return obs, reward, gameover, {} 
             
    def reset(self):
        self.epoch += 1
        self._reset_agent()
        self._reset_apple()
        self._reset_bombs()
        obs = self._get_obs()
        return obs
        
    def render(self,mode="human"):
        obs = self._get_obs().astype(np.uint8)[0].transpose(1,2,0)
        cv2.imshow("game",obs)
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
    hpt = HARGS_T("../hargs/dodgeball2/0000baseline.yaml")
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

