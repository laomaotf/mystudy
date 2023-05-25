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
import fire
from hargs import HARGS

def copy2zip(indir, exts, files,outpath):
    fd = ZipFile(outpath,"a")
    for rdir, _, names in os.walk(indir):
        paths = [os.path.join(rdir,name) for name in names if os.path.splitext(name)[-1] in exts]
        for path in paths:
            rpath = os.path.relpath(path,indir)
            fd.write(path,rpath)
    for path in files:
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
    def __init__(self,input_shape, action_dims,hparam) -> None:
        super().__init__()
        if len(input_shape) == 1:
            input_dims = input_shape[0]
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
        elif len(input_shape) == 3: 
            C,H,W = input_shape
            base_channel = hparam.get("channel_base",8)
            actor_cnn = []
            last_out_ch = base_channel
            for n,name in enumerate(hparam.get("backbone")):
                if name.split(',')[0] == "conv":
                    kernel,stride,padding = [int(n) for n in name.split(',')[1:]]
                    if n == 0:
                        actor_cnn.append(torch.nn.Conv2d(C,base_channel,kernel_size=kernel,
                                                         stride=stride,padding=padding))
                        last_out_ch = base_channel
                    else:
                        actor_cnn.append(torch.nn.Conv2d(last_out_ch,last_out_ch*2,kernel_size=kernel,
                                                         stride=stride,padding=padding))
                        last_out_ch = 2 * last_out_ch
                elif name == 'relu':
                    actor_cnn.append(torch.nn.ReLU())
                elif name == "tanh":
                    actor_cnn.append(torch.nn.Tanh())
                elif name == 'adaptivemaxpool':
                    actor_cnn.append(torch.nn.AdaptiveMaxPool2d((1,1)))
                else: 
                    assert(False),"not supported layer: {}".foramt(name)
            self.actor_conv = torch.nn.Sequential(*actor_cnn)
            # with torch.no_grad():
            #     flatten_dim = self.actor_conv(torch.zeros(1,C,H,W)).shape[1]
            # self.actor_fc = torch.nn.Sequential(
            #     torch.nn.Linear(flatten_dim, action_dims),
            # )
            self.actor_fc = torch.nn.Sequential(
                torch.nn.Linear(last_out_ch, action_dims),
            )
            print(self.actor_conv)
            
            critic_cnn = []
            last_out_ch = base_channel
            for n,name in enumerate(hparam.get("backbone")):
                if name.split(',')[0] == "conv":
                    kernel,stride,padding = [int(n) for n in name.split(',')[1:]]
                    if n == 0:
                        critic_cnn.append(torch.nn.Conv2d(C,base_channel,kernel_size=kernel,
                                                         stride=stride,padding=padding))
                        last_out_ch = base_channel
                    else:
                        critic_cnn.append(torch.nn.Conv2d(last_out_ch,last_out_ch*2,kernel_size=kernel,
                                                         stride=stride,padding=padding))
                        last_out_ch = 2 * last_out_ch
                elif name == 'relu':
                    critic_cnn.append(torch.nn.ReLU())
                elif name == "tanh":
                    critic_cnn.append(torch.nn.Tanh())
                elif name == 'adaptivemaxpool':
                    critic_cnn.append(torch.nn.AdaptiveMaxPool2d((1,1)))    
                else: 
                    assert(False),"not supported layer: {}".foramt(name)
  
            self.critic_conv = torch.nn.Sequential(*critic_cnn)
            print(self.critic_conv)
            # with torch.no_grad():
            #     flatten_dim = self.critic_conv(torch.zeros((1,C,H,W))).shape[-1]
            # self.critic_fc = torch.nn.Sequential(
            #     torch.nn.Linear(flatten_dim,1),
            # )
            self.critic_fc = torch.nn.Sequential(
                torch.nn.Linear(last_out_ch,1),
            )
            
            for m in self.actor_conv:
                if isinstance(m, (torch.nn.Linear, torch.nn.Conv2d)):
                    torch.nn.init.orthogonal_(m.weight,gain=0.5)
                    if m.bias is not None:
                        m.bias.data.fill_(0)
            for m in self.actor_fc:
                if isinstance(m, (torch.nn.Linear, torch.nn.Conv2d)):
                    torch.nn.init.orthogonal_(m.weight,gain=0.5)
                    if m.bias is not None:
                        m.bias.data.fill_(0)
            for m in self.critic_conv:
                if isinstance(m,(torch.nn.Linear, torch.nn.Conv2d)):
                    torch.nn.init.orthogonal_(m.weight,gain=0.5)
                    if m.bias is not None:
                        m.bias.data.fill_(0)
            for m in self.critic_fc:
                if isinstance(m,(torch.nn.Linear, torch.nn.Conv2d)):
                    torch.nn.init.orthogonal_(m.weight,gain=0.5)
                    if m.bias is not None:
                        m.bias.data.fill_(0)
        else:
            assert(False),"input shape is not supported!"
        return
        
    def forward(self,x):
        cx, ax = self.critic_conv(x),self.actor_conv(x)
        cx, ax = cx.view(cx.size()[0],-1), ax.view(ax.size()[0],-1)
        return self.critic_fc(cx), self.actor_fc(ax)
    

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
            self.data['action_prob'].append(action_prob[0,action].numpy()) #batch size = 1
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
        obs = torch.cat(obs,dim=0)
        adv = torch.cat([torch.reshape(x,(1,-1)) for x in adv],dim=0)
        log_prob = torch.cat([torch.reshape(x,(1,-1)) for x in log_prob],dim=0)
        gain = torch.cat([torch.reshape(x,(1,-1)) for x in gain],dim=0)
        return obs,adv,gain,log_prob,action

def main_one(yaml_file):
    hparam = HARGS(yaml_file)
    if hparam.get("use_rand_seed",0):
        np.random.seed(224)  
        torch.manual_seed(224)

    device = torch.device("cuda:0")
    env = dodgeball.CEnv(hparam)
    dataset = PPODataset(env)
    obs_space= env.observation_space
    action_space = env.action_space
    policy = ACPolicy(input_shape = obs_space.shape,action_dims=action_space.n, hparam = hparam)
    policy.to(device)
   
    loss_mse = torch.nn.MSELoss()
    
    
     
    batch_size = 128
    total_epochs = hparam.get("total_epochs",50)
    iters_each_epoch = 1000
    steps_each_collect = 512
    
    
    #optim = torch.optim.SGD(policy.parameters(),lr=0.001,momentum=0.9, weight_decay=1e-5)
    optim = torch.optim.Adam(policy.parameters(),lr=0.0003,eps=1e-5)
    #lr = torch.optim.lr_scheduler.CosineAnnealingLR(optim,eta_min=1e-9,last_epoch=-1, T_max=total_epochs * iters_each_epoch)
    workdir = os.path.dirname(os.path.abspath(__file__))
    hpname = os.path.splitext(os.path.basename(yaml_file))[0]
    outdir = os.path.join(workdir,"exp","{}_{}".format(time.strftime("%Y%m%d_%H%M",time.localtime()),hpname))
    os.makedirs(outdir,exist_ok=True)
    copy2zip(workdir,{".py"}, [yaml_file],  os.path.join(outdir,"codes.zip"))
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
            loss_adv = -1 * torch.min(torch.clamp(ratio,1-0.2, 1 + 0.2) * advantage,ratio*advantage).mean()

            
            #3--loss log_prob(negative entropy. do not push entropy to zero)
            ent = -(log_prob_all * torch.exp(log_prob_all)).sum(dim=-1).mean()
            loss_ent = -ent
            
            loss = loss_adv * hparam.get('adv_loss_weight',1.0) + hparam.get('value_loss_weight') * loss_value \
                + hparam.get('ent_loss_weight',0) * loss_ent
            
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
         
def main(files_or_dir):
    if files_or_dir[0] == '.':
        files_or_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),files_or_dir)
    hparam_files = []
    assert(os.path.exists(files_or_dir)), "{} must exist".format(files_or_dir)
    if os.path.isfile(files_or_dir) and os.path.splitext(files_or_dir)[-1].lower() in {'.yaml','.yml'}:
        hparam_files.append(files_or_dir)
    elif os.path.isdir(files_or_dir):
        hparam_files = [os.path.join(files_or_dir,f) for f in os.listdir(files_or_dir) \
            if os.path.splitext(f)[-1].lower() in {'.yml','.yaml'}]
    else:
        print("bad intpu {}".format(files_or_dir))
    print(hparam_files)
    for hparam_file in hparam_files:
        main_one(hparam_file)
if __name__ == "__main__":
    #main("./hargs/0001c3d1b.yaml")
    fire.Fire(main)
            
              