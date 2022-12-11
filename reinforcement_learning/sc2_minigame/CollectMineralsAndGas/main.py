import gym
from gym import spaces
import numpy as np 
import json
import time
import cv2
import warnings
import sys,os
import subprocess
import shutil
from info import INFO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.env_checker import check_env
import argparse
import pandas as pd


parser = argparse.ArgumentParser(description="start train/test")
parser.add_argument("what",help="train/test")
parser.add_argument("-r","--rew_type",help="gatherMax/leftMax",default="gatherMax")
parser.add_argument("-m","--model",help="model to load",default="")
parser.add_argument("-round_test",help="total round in test",default=10)
parser.add_argument("-output",help="output directory",default="output")
parser.add_argument("-output_tf",help="tensorboard output",default="tflog")

parser.add_argument("-step_per_env",help="step of each env for one unroll",default=512,type=int)
parser.add_argument("-step_total",help="total step in all training",default=100000,type=int)
parser.add_argument("-epoch_train_net",help="epoch when train net",default=10,type=int)
parser.add_argument("-batchsize_train_net",help="batch size when train net",default=64,type=int)

class SC2Env(gym.Env):
    metadata = {"render_modes":["human"],"render_fps":4}
    def __init__(self,reward_type = "gatherMax",name="SC2Env"):
        super().__init__()
        self.actions = ["idle","build_SCV","build_Supplydepot"]
        #self.actions = ["idle"]
        self.name = name
        self.reward_type = reward_type
        # name of observation_space / action_space is determined by check_env()
        # observation_space is input of alg
        # action_space is output of alg
        self.observation_space = spaces.Box(low=0,high=10000,shape=(1,4),dtype=np.int32)
        self.action_space = spaces.Discrete(len(self.actions))
    def step(self, action):
        info = INFO(__file__)
        info.read()
        reward = info.reward()
        iteration = info.iteration()
        supply_workers = info.supply_workers()
        supply_cap = info.supply_cap()
        supply_left = info.supply_left()
        game_over = info.game_over()
        obs = np.asarray([iteration, supply_workers, supply_cap, supply_left])
        #if game_over:
        #    reward = info.map_minerals_total() - info.map_minerals_left()
        #    print(f"#######game over with reward {reward}")
        
        info.rew_type(self.reward_type)
        info.action(self.actions[action])
        info.write()
        
        return np.reshape(obs,(1,4)), reward, game_over, {}
         
    def reset(self):
        info = INFO(__file__)
        info.rew_type(self.reward_type)
        info.action()
        info.write()
        subprocess.Popen(["python", "agent.py"])            
        return np.reshape(np.asarray([-1,0,0,0]),(1,4))
    
    def render(self,mode="human"):
        if mode != "human":
            raise Exception(f"mode must be human")
        return
    #def close(self):
        #cv2.destroyAllWindows()

def is_env_ok():
    env = SC2Env()
    env.reset()
    check_env(env)
    
def test_env():
    env = SC2Env()
    env.reset()
    for _ in range(1000):
        for act in range(len(env.actions)):
            env.step(act)
            env.render()
    env.close()


  
def train_model(args):
    tm = time.strftime("%d%H%M",time.gmtime())
    args.output = os.path.join(args.output,f"train_model_{tm}")
    os.makedirs(args.output,exist_ok=True)
    with open(os.path.join(args.output,"input.json"),'w') as f:
        json.dump(args.__dict__,f,indent=4)
    
    callback_save = CheckpointCallback(save_freq=2000,save_path=args.output,name_prefix="ppo",save_replay_buffer=False)
    env = SC2Env()
    model = PPO("MlpPolicy",env,verbose=1,n_steps=args.step_per_env,tensorboard_log=args.output_tf,
                n_epochs=args.epoch_train_net,batch_size=args.batchsize_train_net,
                _init_setup_model=True)
    #if args.model != "":
    #    model.load(args.model)
    model.learn(total_timesteps=args.step_total * args.step_per_env,reset_num_timesteps=True,
                eval_log_path=1000,
                tb_log_name = "ppo", 
                callback=callback_save)
    model.save(os.path.join(args.output,"final"))
    env.close()
    return

def play_model(args):
    tm = time.strftime("%d%H%M",time.gmtime())
    args.output = os.path.join(args.output,f"play_model_{tm}")
    os.makedirs(args.output,exist_ok=True)
    if args.model != "":
        model = PPO.load(os.path.splitext(args.model)[0])
    else:
        warnings.error("must set model path")
        return 
    with open(os.path.join(args.output,"input.json"),'w') as f:
        json.dump(args.__dict__,f,indent=4)
    results_all = []
    for round  in range(args.round_test):
        env = SC2Env()
        obs = env.reset()
        rewards_epoch = []
        while True:
            action, _states = model.predict(obs)
            obs, rewards, dones, info = env.step(action)
            env.render()  
            rewards_epoch.append(rewards)
            if dones:
                break
        env.close()
        results_all.append(sum(rewards_epoch))
    df = pd.DataFrame({"epoch":[k+1 for k in range(len(results_all))], "rewards":results_all})
    df.to_csv(os.path.join(args.output,"test.csv"),sep=',',index=False,header={"epoch","rewards"})
    
if __name__ == "__main__":
    args = parser.parse_args()
    if args.what == "train":
        train_model(args)
    else:
        play_model(args)