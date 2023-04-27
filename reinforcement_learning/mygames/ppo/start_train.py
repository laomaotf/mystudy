import os,sys
import torch
import gym
import numpy as np
from envs import dodgeball
import torch.functional as F
import copy
import time
from zipfile import ZipFile
from rich.progress import track
from torch.distributions.categorical import Categorical
from torch.utils.tensorboard import SummaryWriter


def copy2zip(indir, exts, outpath):
    fd = ZipFile(outpath,"a")
    for rdir, _, names in os.walk(indir):
        paths = [os.path.join(rdir,name) for name in names if os.path.splitext(name)[-1] in exts]
        for path in paths:
            rpath = os.path.relpath(path,indir)
            fd.write(path,rpath)
    fd.close()
    return

def to_numpy(data,dtype=np.float32):
    if isinstance(data, (list, tuple)):
        data = np.array(data,dtype=dtype)
    elif isinstance(data, (int,float,np.float32)):
        data = np.array(data,dtype=dtype).reshape((1,-1))
    
    if not isinstance(data, np.ndarray):
        raise Exception("can not convert {} to numpy".format(data.dtype))   
    return data

def to_torch(data):
    data = to_numpy(data)
    if isinstance(data, np.ndarray):
        data = torch.from_numpy(data)
    else:
        raise Exception("can not convert {} to torch array".format(data.type()))
    return data
        

class ACPolicy(torch.nn.Module):
    def __init__(self,input_dims, action_dims) -> None:
        super().__init__()
        self.latent_dim = 64
        self.actor = torch.nn.Sequential(
            torch.nn.Linear(input_dims, 64),
            torch.nn.Tanh(),
            torch.nn.Linear(64, self.latent_dim),
            torch.nn.Tanh(),
            torch.nn.Linear(self.latent_dim, action_dims),
        )
        self.critic = torch.nn.Sequential(
            torch.nn.Linear(input_dims, 64),
            torch.nn.Tanh(),
            torch.nn.Linear(64, self.latent_dim),
            torch.nn.Tanh(),
            torch.nn.Linear(self.latent_dim,1),
        )
        
        for m in self.actor:
            if isinstance(m, (torch.nn.Linear, torch.nn.Conv2d)):
                torch.nn.init.orthogonal_(m.weight,gain=0.5)
                if m.bias is not None:
                    m.bias.data.fill_(0)
        for m in self.critic:
            if isinstance(m,(torch.nn.Linear, torch.nn.Conv2d)):
                torch.nn.init.orthogonal_(m.weight,gain=0.5)
                if m.bias is not None:
                    m.bias.data.fill_(0)
        
        return
        
    def forward(self,x):
        return self.critic(x),self.actor(x) 
    

def logist2logsoftmax(logist):
    if isinstance(logist,torch.Tensor):
        log_logist = logist - logist.logsumexp(dim=-1,keepdim=True)
    elif isinstance(logist, np.ndarray):
        log_logist = logist - np.log( np.exp(logist).sum(axis=-1,keepdims=True) )
    return log_logist

def logist2softmax(logist):
    if isinstance(logist,torch.Tensor):
        prob = torch.softmax(logist,dim=-1)
    elif isinstance(logist,np.ndarray):
        prob = np.exp(prob) / np.exp(prob).sum(axis=-1)
    return prob 
    
class PPODataset(object):
    def __init__(self, env) -> None:
        self.env = env
        self.data = {
            "obs":[],
            "reward":[],
            "value":[],
            "action":[],
            "action_prob":[],
            "gameover":[],
            "advantage":[],
            "value":[],
            "gain":[]
        }
        return
    def clear_data(self):
        self.data = {
            "obs":[],
            "reward":[],
            "value":[],
            "action":[],
            "action_prob":[],
            "gameover":[],
            "advantage":[],
            "value":[],
            "gain":[]
        }
        return
    def collect_data(self, total_step, obs, policy,device):
        self.clear_data()
        if isinstance(obs, np.ndarray):
            obs = torch.from_numpy(obs)
        last_obs = obs.to(device)
        policy.eval()
        for step in track(range(total_step),description="collect data..."):
            with torch.no_grad():
                value,action_logist = policy(last_obs)
            value, action_logist = value.cpu(), action_logist.cpu()
            action = Categorical(logits = action_logist).sample()
            action_prob = logist2softmax(action_logist)
            obs,reward,gameover,info = self.env.step(action)
            self.data["obs"].append(to_numpy(last_obs.cpu().detach().numpy()))
            self.data['reward'].append(to_numpy(reward))
            self.data['action'].append(action.numpy())
            self.data['action_prob'].append(action_prob[action].numpy())
            self.data['value'].append(value.numpy())
            self.data['gameover'].append(1 if gameover else 0)
            last_obs = to_torch(obs).to(device)
            if gameover:
                obs = self.env.reset()
                last_obs = torch.from_numpy(obs).to(device)
        self._calc_advantage()   
        return
    def _calc_advantage(self, gamma=0.9, gamma_gae = 0.9):
        total = len(self.data['obs'])
        advantages, gains = [], []
        #####################################################
        #TD(1): one-step forward!!!!!!!!!!!!!!!!!!!!!!!
        delta_next = 0
        for n in range(total-2, -1, -1):
            #if self.data['gameover'][n+1] == 0: !!!!!bug
            value_next = self.data['value'][n+1]
            reward = self.data['reward'][n] 
            gameover = self.data['gameover'][n]
            value = self.data['value'][n]
            delta = reward + value_next * gamma * (1-gameover) - value
            delta_next = delta + delta_next * gamma * gamma_gae * (1 - gameover)
            gains.append(value + delta_next)
            advantages.append(delta_next)
        advantages.reverse()
        gains.reverse()
        self.data['advantage'] = to_numpy(advantages)
        self.data['gain'] = to_numpy(gains)
        
        #strip the last one sample
        for key in self.data:
            if key == "advantage" or key == 'gain':
                continue
            self.data[key]  = self.data[key][0:-1]
        return
    def get_data(self, total):
        indices = np.random.choice([k for k in range(len(self.data['obs']))],total)
        obs, adv, gain, log_prob,action = [],[],[],[],[]
        for i in indices:
            obs.append(to_torch(self.data['obs'][i]))
            adv.append(to_torch(self.data['advantage'][i]))
            gain.append(to_torch(self.data['gain'][i]))
            action_val = self.data['action'][i].item()
            action.append(action_val)
            action_prob = to_torch(self.data['action_prob'][i])
            ###################################
            #log_prob is wrt action !
            log_prob.append(torch.log(action_prob))
        obs = torch.cat([torch.reshape(x,(1,-1)) for x in obs],dim=0)
        adv = torch.cat([torch.reshape(x,(1,-1)) for x in adv],dim=0)
        log_prob = torch.cat([torch.reshape(x,(1,-1)) for x in log_prob],dim=0)
        gain = torch.cat([torch.reshape(x,(1,-1)) for x in gain],dim=0)
        return obs,adv,gain,log_prob,action

def main():
    device = torch.device("cuda:0")
    env = dodgeball.CEnv()
    dataset = PPODataset(env)
    obs_space= env.observation_space
    action_space = env.action_space
    policy = ACPolicy(input_dims = obs_space.shape[0],action_dims=action_space.n)
    policy.to(device)
   
    loss_mse = torch.nn.MSELoss()
    
    
     
    batch_size = 128
    total_epochs = 100
    iters_each_epoch = 1000
    steps_each_collect = 512
    
    
    #optim = torch.optim.SGD(policy.parameters(),lr=0.001,momentum=0.9, weight_decay=1e-5)
    optim = torch.optim.Adam(policy.parameters(),lr=0.0003,eps=1e-5)
    #lr = torch.optim.lr_scheduler.CosineAnnealingLR(optim,eta_min=1e-9,last_epoch=-1, T_max=total_epochs * iters_each_epoch)
    workdir = os.path.dirname(os.path.abspath(__file__))
    outdir = os.path.join(workdir,"exp",time.strftime("%Y%m%d_%H%M",time.localtime()))
    os.makedirs(outdir,exist_ok=True)
    copy2zip(workdir,{".py"}, os.path.join(outdir,"codes.zip"))
    writer = SummaryWriter(outdir)
    step_num = 0
    for epoch in range(total_epochs):
        obs = env.reset()
        print(f"[{epoch+1}]start collect_data... ")
        dataset.collect_data(steps_each_collect,obs=obs,policy=policy,device=device)
        print(f"[{epoch+1}]start training ... ")
        policy.train(True)
        for _ in track(range(iters_each_epoch),description="training..."):
            step_num += 1
            obs, advantage, gains_gt,log_prob_gt,action_gt = dataset.get_data(batch_size)
            obs = obs.to(device)
            value, action_logist = policy(obs)
            #################################
            #log_prob is wrt action
            log_prob_all = logist2logsoftmax(action_logist)
            log_prob = log_prob_all[range(0,batch_size),action_gt].reshape(-1,1)
            #1--loss value
            gains_gt = gains_gt.to(device)
            loss_value = loss_mse(value, gains_gt)
            
            #2--loss advantage
            log_prob_gt = log_prob_gt.to(device)
            advantage = advantage.to(device)
            if advantage.shape[0] > 1:
                advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-8)
            ratio = torch.exp(log_prob - log_prob_gt)
            if 0:
                ratio = torch.min(torch.clamp(ratio,1-0.2, 1 + 0.2),ratio)
                loss_adv = -(ratio * advantage).mean()
            else:
                loss_adv = -1 * torch.min(torch.clamp(ratio,1-0.2, 1 + 0.2) * advantage,ratio*advantage).mean()

            
            #3--loss log_prob(negative entropy. do not push entropy to zero)
            ent = -(log_prob_all * torch.exp(log_prob_all)).sum(dim=-1).mean()
            loss_ent = -ent
            
            loss = 1 * (loss_adv + 0.5 * loss_value + 0.00 * loss_ent)
            
            optim.zero_grad()
            loss.backward()
            ####################################
            #clip gradients
            torch.nn.utils.clip_grad_norm_(policy.parameters(),0.5)
            optim.step()
            #lr.step()
           

            writer.add_scalar("loss",loss.item(),step_num)
            writer.add_scalar("loss_value",loss_value.item(),step_num)
            writer.add_scalar("loss_ent",loss_ent.item(),step_num)
            writer.add_scalar("loss_adv",loss_adv.item(),step_num)
            #writer.add_scalar("lr",lr.get_last_lr()[0],step_num)
            
        print(f"[{epoch+1}] start eval...")
        torch.save(policy.state_dict(),os.path.join(outdir,"{}.pth".format(step_num)))
        policy.eval() 
        lifetimes, rewards = [], []
        for _ in track(range(3),description="eval..."):
            last_obs = env.reset()
            lifetime_sum,reward_sum = 0,0 
            while True:
                last_obs = to_torch(last_obs).to(device)
                _,action_logist = policy(last_obs)
                action = torch.argmax(action_logist).cpu().item()  
                last_obs,reward,gameover,info = env.step(action)  
                if gameover:
                    break
                lifetime_sum += 1
                reward_sum += reward
            lifetimes.append(lifetime_sum)
            rewards.append(reward_sum)
        writer.add_scalar("lifetime",np.mean(lifetimes),step_num)
        writer.add_scalar("reward",np.mean(rewards),step_num)
                 
    writer.close()         
         

if __name__ == "__main__":
    main() 
            
              