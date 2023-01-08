import gym
from gym import spaces
import numpy as np 
import json,copy
import time
import cv2
import warnings
import sys,os
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3 import PPO
import argparse
from envs import walk_in_line,baseline,dodgeball

parser = argparse.ArgumentParser(description="start train/test")
parser.add_argument("what",help="train/play")
parser.add_argument("env",help="env name")
parser.add_argument("-m","--model",help="model to load",default="")
parser.add_argument("-round_test",help="total round in test",default=10)
parser.add_argument("-output",help="output directory",default="output")

parser.add_argument("-step_per_env",help="step of each env for one unroll",default=512,type=int)
parser.add_argument("-step_total",help="total step in all training",default=100000,type=int)
parser.add_argument("-epoch_train_net",help="epoch when train net",default=10,type=int)
parser.add_argument("-batchsize_train_net",help="batch size when train net",default=64,type=int)

 
def train_model(args):
    tm = time.strftime("%d%H%M",time.gmtime())
    args.output = os.path.join(args.output,f"train_model_{tm}")
    os.makedirs(args.output,exist_ok=True)
    with open(os.path.join(args.output,"input.json"),'w') as f:
        json.dump(args.__dict__,f,indent=4)
    
    callback_save = CheckpointCallback(save_freq=2000,save_path=args.output,name_prefix="ppo",save_replay_buffer=False)
    env = eval(f"{args.env}.CEnv()")
    model = PPO("MlpPolicy",env,verbose=1,n_steps=args.step_per_env,tensorboard_log=args.output,
                n_epochs=args.epoch_train_net,batch_size=args.batchsize_train_net,
                _init_setup_model=True, policy_kwargs = { "net_arch" : [dict(pi=[128//2, 128//2], vf=[128//2, 128//2])]})
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
    steps_total = []
    env = eval(f"{args.env}.CEnv()")
    for round  in range(args.round_test):
        obs = env.reset()
        steps = 0
        while True:
            action, _states = model.predict(obs)
            obs, rewards, dones, info = env.step(action)
            env.render()  
            steps += 1
            if dones:
                break
        steps_total.append(steps)
        env.close()
    steps_mean = np.mean(steps_total)
    steps_std = np.std(steps_total)
    print("mean/std steps of {}: {} / {} ".format(args.round_test, steps_mean, steps_std))
  
if __name__ == "__main__":
    args = parser.parse_args()
    if args.what == "train":
        train_model(args)
    elif args.what == "play":
        play_model(args)