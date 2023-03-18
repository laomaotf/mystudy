import random,os
import logging,time
import numpy as np
import json
import cv2
import copy
import datetime
from rich.progress import track
import multiprocessing as mp
import pickle
import shutil
from torch.utils.tensorboard import SummaryWriter

MAKE_WORLD_EASY = False

VISUAL_SCALE = 10

# Defining the population size.

VIEW_SIZE = 5

GENERATION_TOTAL = 3000

WORLD_SIZE = 50
FOOD_CAPACITY = int(WORLD_SIZE / 2)
DAYTIME = WORLD_SIZE * 3
#AGENT_IN_DIM = FOOD_CAPACITY * 2 + 1 + 1 + 2 #FOOD(x,y), DAYTIME,reward position

AGENT_IN_DIM =  2*(VIEW_SIZE*2+1)**2 + 2 + 1
AGENT_OUT_DIM = 2 #dx,dy


POPULATION_SIZE = FOOD_CAPACITY

HIDDEN_DIM = (AGENT_IN_DIM + AGENT_OUT_DIM) // 2  + AGENT_OUT_DIM
AGENT_WEIGHT_DIM = AGENT_IN_DIM * HIDDEN_DIM + HIDDEN_DIM * AGENT_OUT_DIM

NUM_WORKER = 4

class GENE:
    def __init__(self,seeds,sigma=0.005):
        np.random.seed(seeds[0])
        self.dna = np.random.uniform(low=-1,high=1,size=(AGENT_WEIGHT_DIM,))
        for seed in seeds[1:]:
            np.random.seed(seed)
            self.dna += sigma * np.random.uniform(low=-1,high=1,size=(AGENT_WEIGHT_DIM,))
        return
    def get_act(self,obs):
        w01 = self.dna[0:AGENT_IN_DIM * HIDDEN_DIM]
        w01 = w01.reshape(HIDDEN_DIM, AGENT_IN_DIM).astype(np.float32)
        w12 = self.dna[HIDDEN_DIM * AGENT_IN_DIM:]
        w12 = w12.reshape(AGENT_OUT_DIM, HIDDEN_DIM).astype(np.float32)
        input = obs.astype(np.float32) 
        hidden = w01 @ input
        hidden = 0.5 * (np.abs(hidden) + hidden)
        dx,dy = w12 @ hidden 
        if dx >= 0:
            dx = 1
        else:
            dx = -1
            
        if dy >= 0:
            dy = 1
        else:
            dy = -1
        return dx, dy

class AGENT(GENE):
    def __init__(self, seeds, sigma=0.005):
        super().__init__(seeds, sigma)
        #using seeds[-1] here
        self.x = random.randint(0, WORLD_SIZE-1)
        self.y = random.randint(0, WORLD_SIZE-1)
        self.rew = 0
    def step(self,obs):
        dx,dy = self.get_act(obs)
        self.x, self.y = int(self.x + dx)%WORLD_SIZE, int(self.y + dy)%WORLD_SIZE
        return
    def add_reward(self,rew):
        self.rew += rew
        return

def visual_world(world,agents,t):
    image_world = (world * 255).astype(np.uint8)
    image_world = cv2.cvtColor(image_world,cv2.COLOR_GRAY2BGR)
    H,W = image_world.shape[0:2]
    image_world = cv2.resize(image_world,(W*VISUAL_SCALE,H*VISUAL_SCALE),interpolation=cv2.INTER_NEAREST)
    for k in range(len(agents)):
        x,y = agents[k].x * VISUAL_SCALE, agents[k].y * VISUAL_SCALE
        rew = agents[k].rew 
        if rew > 0:
            cv2.circle(image_world,(x,y),VISUAL_SCALE//2,(0,255,0),1)
        else:
            cv2.circle(image_world,(x,y),VISUAL_SCALE//2,(0,255,255),1)
    cv2.putText(image_world,f"{t}",(20,20),cv2.FONT_HERSHEY_COMPLEX,1.5,(128,128,128))
    cv2.imshow("world",image_world)
    cv2.waitKey(1) 
    
def get_obs_local(world,agents,index_anchor,t):
    ax,ay = agents[index_anchor].x, agents[index_anchor].y
    obs_world = []
    for y in range(ay - VIEW_SIZE, ay+VIEW_SIZE+1):
        for x in range(ax - VIEW_SIZE, ax + VIEW_SIZE+1):
            nx,ny = x%WORLD_SIZE, y%WORLD_SIZE
            obs_world.append( world[ny,nx] )
    obs_others = np.zeros((2*VIEW_SIZE+1,2*VIEW_SIZE+1))
    for k in range(len(agents)):
        dx,dy = agents[k].x - ax, agents[k].y - ay
        if abs(dx) > VIEW_SIZE or abs(dy) > VIEW_SIZE: 
            continue
        obs_others[dy+VIEW_SIZE,dx+VIEW_SIZE] += 1 #count number of other players
    obs_others = obs_others.flatten().tolist()
    return obs_world + obs_others + [ax,ay,t]
        
        
        
            
    
        
def play_one_epoch(epoch,pops,seed,flag_visual):
    agents = []
    for pop in pops:
        agent = AGENT(pop)
        agents.append(agent)
    random.seed(seed)
    agents_num = len(agents)
    world = [1 for _ in range(FOOD_CAPACITY)] + [0 for _ in range(WORLD_SIZE*WORLD_SIZE - FOOD_CAPACITY)]
    if MAKE_WORLD_EASY:
        random.seed(42) ###########################################
    random.shuffle(world)
    keep_living = np.zeros(agents_num)
    world = np.array(world,dtype=np.int32).reshape((WORLD_SIZE,WORLD_SIZE))
    for t in range(DAYTIME-1,0,-1):
        for agt_k in range(agents_num):
            if 0: #global obs
                xy_food = np.array(np.argwhere(world == 1)).reshape((-1,2)).flatten().tolist()
                if len(xy_food) < 2 * FOOD_CAPACITY:
                    xy_food = xy_food + [-1 for _ in range(2 * FOOD_CAPACITY - len(xy_food))]
                xy_agent = np.array([agents[agt_k].x, agents[agt_k].y]).reshape((1,2)).flatten().tolist()
                obs = np.array(xy_food + [t] + [agents[agt_k].rew] + xy_agent).reshape((-1,1))
            else:
                obs = get_obs_local(world,agents,agt_k,t)
                obs = np.array(obs).reshape((-1,1)) 
            agents[agt_k].step(obs)
            agents[agt_k].add_reward( world[agents[agt_k].y, agents[agt_k].x]  )
            world[agents[agt_k].y, agents[agt_k].x] = 0 #eat food if exist
        if flag_visual and epoch < 2:
            visual_world(world,agents,t)
            
    for agt_k in range(agents_num):
        if agents[agt_k].rew > 0 and agents[agt_k].x * agents[agt_k].y == 0:
            keep_living[agt_k] += 1
    return epoch,keep_living  

def play_epochs(args):
    pid,epochs,pops = args
    results = []
    for epoch in epochs:
        seed = int(time.time() + pid)
        result = play_one_epoch(epoch,pops,seed,pid==0)
        results.append(result)
    with open(f"{pid}.pkl",'wb') as f:
        pickle.dump(results,f)
    
def calc_fitness_fast(agents,epoch_total = 100):
    pool = mp.Pool(NUM_WORKER)
    params = []
    epoch_step = epoch_total // NUM_WORKER
    pids = []
    for pid,epoch in enumerate(range(0,epoch_total,epoch_step)):
        pids.append(pid) 
        path = f"{pid}.pkl"
        if os.path.exists(path):
            os.remove(path)
        params.append((pid,[k for k in range(epoch,epoch+epoch_step,1)],agents))
    if NUM_WORKER == 1:
        play_epochs(*params)
    else:
        pool.map(play_epochs,params)
        pool.close()
        pool.join()
    results = []
    for pid in pids:
        with open(f"{pid}.pkl", 'rb') as f:
            res = pickle.load(f)
            results.extend(res)
    epochs_keep_living = np.zeros(len(agents))
    for _,res in results:
        epochs_keep_living += res
    return (epochs_keep_living / len(results)).tolist()
    
def calc_fitness(agents,epoch_total = 100):
    return calc_fitness_fast(agents,epoch_total=epoch_total)

 
def select_survivors(pops, fitness, num_survivors, num_genius):
    survivors, geniuses = [],[]
    fitness_survivors, fitness_geniuses = [], []
    indices = np.argsort(fitness)[::-1]
    for n in range(num_genius):
        i = indices[n]
        geniuses.append(copy.deepcopy(pops[i]))
        fitness_geniuses.append(fitness[i])
    for n in range(num_survivors):
        i = indices[n]
        survivors.append(copy.deepcopy(pops[i]))
        fitness_survivors.append(fitness[i])
    return geniuses,fitness_geniuses, survivors, fitness_survivors

def new_generation(pops,num):
    indices = np.random.choice(range(len(pops)),size=num)
    offsprings = []
    for i in indices:
        pop = copy.deepcopy(pops[i]) + [random.randrange(0,2**31)]
        offsprings.append(pop)
    return offsprings

def saveGeneration(path, pops, fitnesses):
    json_content = {"pops":pops, "fitness":fitnesses}
    with open(path.format(path),'w') as f:
        json.dump(json_content,f)
    return

def loadGeneration(path):
    with open(path,"r") as f:
        json_content = json.load(f)
    pops = json_content['pops'] 
    return pops 

def main(): 
    now = datetime.datetime.now()
    outdir = os.path.join("output","{}{}{}_{}-{}".format(now.year,now.month,now.day,now.hour,now.minute))
    os.makedirs(outdir,exist_ok=True) 
    shutil.copy(__file__,outdir)
    writer = SummaryWriter(log_dir=os.path.join(outdir,"tblog"),flush_secs=120)
    if os.path.exists("pretrained.json"):
        pops = loadGeneration("pretrained.json")
        print("start with pretrained weights")
    else:
        pops = []
        np.random.seed(int(time.time()))
        for _ in range(POPULATION_SIZE):
            pops.append([random.randint(0,2**31)])
        print("start from scratch")
    geniuses_last = {
        "pops":[],
        "fitness":[]
    }
    startT = time.time()
    for generation in range(GENERATION_TOTAL):
        fitness = calc_fitness(pops)
        T = time.time() - startT
        geniuses, fitness_geniuses, survivors,fitness_survivors = select_survivors(pops + geniuses_last['pops'],
                                                                             fitness + geniuses_last['fitness'],
                                                                             POPULATION_SIZE//4,2)
        pops = new_generation(survivors,POPULATION_SIZE)
        geniuses_last['pops'], geniuses_last['fitness'] = geniuses, fitness_geniuses   
        max_fitness = np.max(fitness_survivors)
        avg_fitness = np.mean(fitness_survivors)
        std_fitness = np.std(fitness_survivors)
        writer.add_scalars(main_tag="fitness",tag_scalar_dict={
            "max":max_fitness, "avg":avg_fitness,"std":std_fitness
        },global_step=generation+1) 
        
        print('genenration:{}, fitness:{:.4f},{:.4f},{:.4f}, {:.2f}H'.format(
            generation+1,max_fitness,avg_fitness, std_fitness,T/3600.0))
        saveGeneration(os.path.join(outdir,"{}_{:.4f}.json".format(generation+1, avg_fitness)), survivors, fitness_survivors)
    T = time.time() - startT
    logging.info("run {} generations within {:.2f}min".format(GENERATION_TOTAL,T/60))
    writer.close()
       

if __name__=="__main__":
    main()
   