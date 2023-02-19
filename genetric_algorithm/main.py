import random,os
import logging,time
import numpy as np
import json
import cv2
from rich.progress import track
import multiprocessing as mp
import pickle

VISUAL_SCALE = 10

# Defining the population size.
POPULATION_SIZE = 200

WEIGHT_MIN, WEIGHT_MAX, WEIGHT_STEP = -1.0,1.0,0.01

GENERATION_TOTAL = 10000
MATING_TOTAL = max([int(POPULATION_SIZE * 0.05),2])

WORLD_SIZE = 50
FOOD_CAPACITY = int(WORLD_SIZE / 2)
DAYTIME = WORLD_SIZE * 3
AGENT_IN_DIM = FOOD_CAPACITY * 2 + 1 + 1 + 2 #FOOD(x,y), DAYTIME,reward position
AGENT_OUT_DIM = 2 #dx,dy

AGENT_WEIGHT_DIM = AGENT_IN_DIM * AGENT_OUT_DIM


def get_action(obs,weights):
    model = weights.reshape((AGENT_OUT_DIM, AGENT_IN_DIM)).astype(np.float32)
    input = obs.astype(np.float32) 
    dx,dy = model @ input
    if dx >= 0:
        dx = 1
    else:
        dx = -1
        
    if dy >= 0:
        dy = 1
    else:
        dy = -1
    return dx, dy

def play_one_epoch(epoch,agents):
    agent_num = len(agents)
    world = [1 for _ in range(FOOD_CAPACITY)] + [0 for _ in range(WORLD_SIZE*WORLD_SIZE - FOOD_CAPACITY)]
    random.shuffle(world)
    keep_living = np.zeros(agent_num)
    world = np.array(world,dtype=np.int32).reshape((WORLD_SIZE,WORLD_SIZE))
    xs_agent = np.random.choice([k for k in range(WORLD_SIZE)],size=agent_num)
    ys_agent = np.random.choice([k for k in range(WORLD_SIZE)],size=agent_num)
    rewards_agent = np.zeros(agent_num)
    for t in range(DAYTIME-1,0,-1):
        for agt_k in range(agent_num):
            xy_food = np.array(np.argwhere(world == 1)).reshape((-1,2)).flatten().tolist()
            if len(xy_food) < 2 * FOOD_CAPACITY:
                xy_food = xy_food + [-1 for _ in range(2 * FOOD_CAPACITY - len(xy_food))]
            xy_agent = np.array([xs_agent[agt_k],ys_agent[agt_k]]).reshape((1,2)).flatten().tolist()
            obs = np.array(xy_food + [t] + [rewards_agent[agt_k]] + xy_agent).reshape((-1,1))
            dx,dy = get_action(obs, agents[agt_k]) 
            xs_agent[agt_k] = int(xs_agent[agt_k] + dx) % WORLD_SIZE
            ys_agent[agt_k] = int(ys_agent[agt_k] + dy) % WORLD_SIZE
            # xs_agent[agt_k] = max([0,int(xs_agent[agt_k] + dx)])
            # ys_agent[agt_k] = max([0,int(ys_agent[agt_k] + dy)])
            # xs_agent[agt_k] = min([WORLD_SIZE-1, xs_agent[agt_k]])
            # ys_agent[agt_k] = min([WORLD_SIZE-1, ys_agent[agt_k]])
            rewards_agent[agt_k] += world[ys_agent[agt_k], xs_agent[agt_k]] 
            world[ys_agent[agt_k], xs_agent[agt_k]] = 0 #eat food if exist
    for agt_k in range(agent_num):
        if rewards_agent[agt_k] > 0 and xs_agent[agt_k] * ys_agent[agt_k] == 0:
            keep_living[agt_k] += 1
    return epoch,keep_living  

def play_epochs(args):
    pid,epochs,agents = args
    results = []
    for epoch in epochs:
        result = play_one_epoch(epoch,agents)
        results.append(result)
    with open(f"{pid}.pkl",'wb') as f:
        pickle.dump(results,f)
    
def calc_fitness_fast(agents,epoch_total = 100):
    thread_num = 4
    pool = mp.Pool(thread_num)
    params = []
    epoch_step = epoch_total // thread_num
    pids = []
    for pid,epoch in enumerate(range(0,epoch_total,epoch_step)):
        pids.append(pid) 
        path = f"{pid}.pkl"
        if os.path.exists(path):
            os.remove(path)
        params.append((pid,[k for k in range(epoch,epoch+epoch_step,1)],agents))
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
    return epochs_keep_living / len(results)
    
def calc_fitness_debug(agents,epoch_total = 100):
    epochs_keep_living = np.zeros(len(agents))
    for epoch in track(range(epoch_total),description="evaulation"):
        world = [1 for _ in range(FOOD_CAPACITY)] + [0 for _ in range(WORLD_SIZE*WORLD_SIZE - FOOD_CAPACITY)]
        random.shuffle(world)
        world = np.array(world,dtype=np.int32).reshape((WORLD_SIZE,WORLD_SIZE))
        xs_agent = np.random.choice([k for k in range(WORLD_SIZE)],size=len(agents))
        ys_agent = np.random.choice([k for k in range(WORLD_SIZE)],size=len(agents))
        rewards_agent = np.zeros(len(agents))
        if VISUAL_SCALE > 0:
            world_vis = (world.copy() * 255).astype(np.uint8)
            world_vis = cv2.cvtColor(world_vis,cv2.COLOR_GRAY2BGR)
            world_vis = cv2.resize(world_vis,(0,0),fx=VISUAL_SCALE, fy = VISUAL_SCALE, interpolation=cv2.INTER_NEAREST)
        for t in range(DAYTIME-1,0,-1):
            for agt_k in range(len(agents)):
                xy_food = np.array(np.argwhere(world == 1)).reshape((-1,2)).flatten().tolist()
                if len(xy_food) < 2 * FOOD_CAPACITY:
                    xy_food = xy_food + [-1 for _ in range(2 * FOOD_CAPACITY - len(xy_food))]
                xy_agent = np.array([xs_agent[agt_k],ys_agent[agt_k]]).reshape((1,2)).flatten().tolist()
                obs = np.array(xy_food + [t] + [rewards_agent[agt_k]] + xy_agent).reshape((-1,1))
                dx,dy = get_action(obs, agents[agt_k]) 
                xs_agent[agt_k] = int(xs_agent[agt_k] + dx) % WORLD_SIZE
                ys_agent[agt_k] = int(ys_agent[agt_k] + dy) % WORLD_SIZE
                # xs_agent[agt_k] = max([0,int(xs_agent[agt_k] + dx)])
                # ys_agent[agt_k] = max([0,int(ys_agent[agt_k] + dy)])
                # xs_agent[agt_k] = min([WORLD_SIZE-1, xs_agent[agt_k]])
                # ys_agent[agt_k] = min([WORLD_SIZE-1, ys_agent[agt_k]])
                rewards_agent[agt_k] += world[ys_agent[agt_k], xs_agent[agt_k]] 
                world[ys_agent[agt_k], xs_agent[agt_k]] = 0 #eat food if exist
                if VISUAL_SCALE > 0 and epoch == epoch_total - 1:
                    world_vis = (world.copy() * 255).astype(np.uint8)
                    world_vis = cv2.cvtColor(world_vis,cv2.COLOR_GRAY2BGR)
                    world_vis = cv2.resize(world_vis,(0,0),fx=VISUAL_SCALE, fy = VISUAL_SCALE, interpolation=cv2.INTER_NEAREST)
                    sorted_idx = np.argsort(rewards_agent)
                    for k in sorted_idx:
                        x,y,r = xs_agent[k],ys_agent[k],rewards_agent[k]
                        ix, iy = int(x * VISUAL_SCALE), int(y * VISUAL_SCALE)
                        cv2.putText(world_vis,f"{t}",(10,50),cv2.FONT_HERSHEY_COMPLEX,1.0,(100,100,100))
                        if r > 0:
                            color = (0,255,0)
                        else:
                            color = (0,0,255)
                        cv2.circle(world_vis, (ix,iy), int(VISUAL_SCALE)//2, color, 2)
                    cv2.imshow("evaulation",world_vis)
                    cv2.waitKey(1)
        if VISUAL_SCALE > 0 and epoch == epoch_total - 1:
            world_vis = (world.copy() * 255).astype(np.uint8)
            world_vis = cv2.cvtColor(world_vis,cv2.COLOR_GRAY2BGR)
            world_vis = cv2.resize(world_vis,(0,0),fx=VISUAL_SCALE, fy = VISUAL_SCALE, interpolation=cv2.INTER_NEAREST)
        for agt_k in range(len(agents)):
            if rewards_agent[agt_k] > 0 and xs_agent[agt_k] * ys_agent[agt_k] == 0:
                epochs_keep_living[agt_k] += 1
                if VISUAL_SCALE > 0 and epoch == epoch_total - 1:
                    x,y = xs_agent[agt_k],ys_agent[agt_k]
                    ix, iy = int(x * VISUAL_SCALE), int(y * VISUAL_SCALE)
                    cv2.circle(world_vis, (ix,iy), VISUAL_SCALE//2, (0,255,0), 2)
            else:
                if VISUAL_SCALE > 0 and epoch == epoch_total - 1:
                    x,y = xs_agent[agt_k],ys_agent[agt_k] 
                    ix, iy = int(x * VISUAL_SCALE), int(y * VISUAL_SCALE)
                    cv2.circle(world_vis, (ix,iy), 1, (128,128,128), 1)
        if VISUAL_SCALE > 0 and epoch == epoch_total - 1:
            cv2.imshow("result",world_vis)
            cv2.waitKey(1)
    if VISUAL_SCALE > 0: 
        cv2.destroyAllWindows()
    return epochs_keep_living / epoch_total
            
def calc_fitness(agents,epoch_total = 100):
    return calc_fitness_debug(agents,epoch_total=epoch_total)
 
def select_mating_pool(pop, fitness, num_parents):
    # Selecting the best individuals in the current generation as parents for producing the offspring of the next generation.
    parents = []
    while True:
        mf = np.max(fitness)
        if mf < 0.0001 or len(parents) >= num_parents:
            break
        max_fitness_idx = np.argwhere(fitness == mf)
        for idx in max_fitness_idx:
            parents.append( pop[idx[0], :].flatten().tolist() )
            fitness[idx[0]] = -99999999
    return np.array(parents).reshape((-1,AGENT_WEIGHT_DIM))

def crossover(parents, offspring_size):
    # creating children for next generation 
    if parents.shape[0] == 1:
        return parents.copy()
    offspring = np.empty(offspring_size)
    for k in range(offspring_size[0]): 
        while True:
            parent1_idx = random.randint(0, parents.shape[0] - 1)
            parent2_idx = random.randint(0, parents.shape[0] - 1)
            # produce offspring from two parents if they are different
            if parent1_idx == parent2_idx:
                continue
            for j in range(offspring_size[1]):
                if random.uniform(0, 1) < 0.5:
                    offspring[k, j] = parents[parent1_idx, j]
                else:
                    offspring[k, j] = parents[parent2_idx, j]
            break
    return offspring


def mutation(offspring_crossover):
    # mutating the offsprings generated from crossover to maintain variation in the population
    
    for idx in range(offspring_crossover.shape[0]):
        for _ in range(AGENT_WEIGHT_DIM): #make mutation for some genes
            random_indice = random.randint(0,offspring_crossover.shape[1]-1)
            random_value = np.random.choice(np.arange(WEIGHT_MIN/10,WEIGHT_MAX/10,step=WEIGHT_STEP/10),size=(1),replace=False)
            offspring_crossover[idx, random_indice] = np.clip(offspring_crossover[idx, random_indice] + random_value,
                                                              WEIGHT_MIN,WEIGHT_MAX)
    offspring_crossover = np.clip(offspring_crossover,WEIGHT_MIN,WEIGHT_MAX)
    return offspring_crossover

def saveGeneration(path, pops, fitnesses):
    json_content = [] 
    pop_num, pop_dim = pops.shape
    json_content.append({"num":pop_num,"dim":pop_dim})
    for pop, fitness in zip(pops, fitnesses):
        pop = pop.tolist()
        json_content.append({"pop":pop, 'fit':fitness})
    with open(path.format(path),'w') as f:
        json.dump(json_content,f)
    return

def main(): 
    outdir = os.path.join("output")
    os.makedirs(outdir,exist_ok=True) 
    new_population = np.random.choice(np.arange(WEIGHT_MIN,WEIGHT_MAX,step=WEIGHT_STEP),size=(POPULATION_SIZE, AGENT_WEIGHT_DIM),replace=True)
    startT = time.time()
    for generation in range(GENERATION_TOTAL):
        # Measuring the fitness of each chromosome in the population.
        fitness = calc_fitness(new_population)
        T = time.time() - startT
        print('generation={}, max fitness={:.4f}, average fitness={:.4f} time elapsed={:.2f}min'.format(generation+1,np.max(fitness),np.mean(fitness),T/60))
        saveGeneration(os.path.join(outdir,"{}_{:.4f}.json".format(generation+1, np.mean(fitness))), new_population, fitness)

        
        # Selecting the best parents in the population for mating.
        parents = select_mating_pool(new_population, fitness, MATING_TOTAL)

        # Generating next generation using crossover.
        offspring_crossover = crossover(parents, offspring_size=(POPULATION_SIZE - parents.shape[0], AGENT_WEIGHT_DIM))

        # Adding some variations to the offsrping using mutation.
        offspring_mutation = mutation(offspring_crossover)

        # Creating the new population based on the parents and offspring.
        new_population[0:parents.shape[0], :] = parents
        new_population[parents.shape[0]:, :] = offspring_mutation
    
    T = time.time() - startT
    logging.info("run {} generations within {:.2f}min".format(GENERATION_TOTAL,T/60))
       

if __name__=="__main__":
    main()
   