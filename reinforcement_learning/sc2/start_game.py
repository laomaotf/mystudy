from ast import parse
from wsgiref.simple_server import make_server
import gym
from gym import spaces
import numpy as np 
import pickle
import time
import cv2
import sys,os
#from gym.utils.env_checker import check_env
from configs.config import CONFIG
import subprocess
import shutil
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.env_checker import check_env
import argparse
from utils import logger

parser = argparse.ArgumentParser(description="start train/test")
parser.add_argument("what",help="train/test")
parser.add_argument("config",help="config file")
parser.add_argument("--model",help="model to load if necessary")

class SC2Env(gym.Env):
    metadata = {"render_modes":["human"],"render_fps":4}
    def __init__(self,config_file,flag_debug=False, name="SC2Env"):
        super().__init__()
        self.cfg = CONFIG(config_file)
        self.config_file = config_file
        self.size = self.cfg.get('observation.size')
        self.actions = self.cfg.get('action')
        self.name = name
        # name of observation_space / action_space is determined by check_env()
        # observation_space is input of alg
        # action_space is output of alg
        self.observation_space = spaces.Box(low=0,high=255,shape=(self.size,self.size,3),dtype=np.uint8)
        self.action_space = spaces.Discrete(len(self.actions))
        self.flag_debug=flag_debug
    def step(self, action):
        #print(f"game::step({action})")
        #send action to agent
        while True:
            try: 
                with open("info.dat",'rb') as f:
                    info = pickle.load(f)
                if info['action'] is None:
                    info['action'] = self.actions[action]
                    with open('info.dat','wb') as f:
                        pickle.dump(info,f)
                    break 
                time.sleep(0.05)
            except Exception as e:
                time.sleep(0.05)
                pass
        #receive agent feedback
        with open('info.dat','rb') as f:
            info = pickle.load(f)
            
        if info['terminated']:
            logger.info(f"GameOver With:{info['result']}")    
        return info['obs'], info['reward'], info['terminated'],info 
                
         
    def reset(self):
        #print("game::reset")
        info = {
            "obs":np.zeros((self.size,self.size,3),dtype=np.uint8),
            "name":self.name,
            "reward":0,
            "action":None,
            "terminated":False,
            "config_file":self.config_file,
            "flag_debug":self.flag_debug,
            "last_patrol":0,
            "result":"None"
        } 
        with open("info.dat",'wb') as f:
            pickle.dump(info,f)
            
        subprocess.Popen(["python", "agentsc2.py", f"{self.config_file}"])            
        return info['obs']
    
    def render(self,mode="human"):
        if mode != "human":
            raise Exception(f"mode must be human")
        with open("info.dat",'rb') as f:
            info = pickle.load(f)
        if info['obs'] is not None:
            vis = cv2.flip(info['obs'],0)
            vis = cv2.resize(vis,(512,512),0,0,interpolation=cv2.INTER_LINEAR)
            cv2.imshow(self.name + "_render",vis)
            cv2.waitKey(1)
        return
    def close(self):
        cv2.destroyAllWindows()


def mkdir_train_output_dir(config_file,root_dir = "train_output"):
    os.makedirs(root_dir,exist_ok=True)
    train_output_existed = list(os.listdir(root_dir))
    if train_output_existed == []:
        train_output_index = 0
    else:
        train_output_index = max([int(x[:3]) for x in train_output_existed]) + 1
    basename = os.path.splitext(os.path.basename(config_file))[0]
    outdir =  os.path .join(root_dir,f"{train_output_index:>03}_" + basename)
    os.makedirs(outdir)
    return outdir

def is_env_ok(config_file):
    env = SC2Env(config_file)
    env.reset()
    check_env(env)
    
def test_env(config_file):
    env = SC2Env(config_file)
    env.reset()
    for _ in range(1000):
        for act in range(len(env.actions)):
            env.step(act)
            env.render()
    env.close()
   
def train_model_with_env(config_file):
    train_output_dir = mkdir_train_output_dir(config_file)
    logger.info(f"train output dir: {train_output_dir}")
    callback_save = CheckpointCallback(save_freq=5000,save_path=train_output_dir,name_prefix="sc2",save_replay_buffer=False)
    shutil.copy(config_file,train_output_dir)
    cfg = CONFIG(config_file)
    if cfg.get("train.algname") == "PPO":
        env = SC2Env(config_file,flag_debug=True)
        model = PPO(cfg("train.network"),env,verbose=1,tensorboard_log=os.path.join(train_output_dir,"tblog"))
        model.learn(total_timesteps=cfg("train.total_timesteps"),tb_log_name="PPO",reset_num_timesteps=False,
                    eval_log_path=os.path.join(train_output_dir,"log"),
                    callback=callback_save)
        model.save(os.path.join(train_output_dir,"pposc2"))
        env.close()
    return
        
    
def test_model_with_env(config_file,model_path,round_total = 10):
    model = PPO.load(os.path.splitext(model_path)[0])
    results_all = []
    for round  in range(round_total):
        env = SC2Env(config_file)
        obs = env.reset()
        while True:
            action, _states = model.predict(obs)
            obs, rewards, dones, info = env.step(action)
            env.render()  
            if dones:
                results_all.append(info['result'])
                break
        env.close()
    print(results_all)

if __name__ == "__main__":
    logger.info("=====================START =================================") 
    args = parser.parse_args()
    logger.info(f"{args}")
    args.config = args.config + ".yaml"
    config_file = os.path.join(os.path.dirname(__file__), "configs", args.config)
    if args.what == "train":
        #test_env(config_file)
        train_model_with_env(config_file)
    elif args.what == "test": 
        model_path = os.path.join(os.path.dirname(__file__), "train_output", args.model)
        test_model_with_env(config_file, model_path)
    else:
        print(f"what is meanings of {args.what}")