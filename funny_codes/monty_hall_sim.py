import numpy as np 
from matplotlib import pyplot as plt 
import random
import time
from matplotlib import animation 
plt.rcParams['font.sans-serif'] = ['SimHei'] # 步骤一（替换sans-serif字体）
plt.rcParams['axes.unicode_minus'] = False  # 步骤二（解决坐标轴负数的负号显示问题）

N_ROUND = 1000


random.seed(time.time())

result_by_first_choose = []
result_if_change_door = []


is_car = lambda x: x == 'car'
for n in range(N_ROUND):
    targets = ["car","goat","goat"]
    random.shuffle(targets)
    doors_all = [0,1,2] 
    first_choose = random.choice(doors_all)
    doors_left = list(set(doors_all) - set([first_choose]))
    random.shuffle(doors_left)
    for door_open in doors_left:
        if targets[door_open] == "goat":
            break
    result_by_first_choose.append(targets[first_choose])
    second_choose = list(set(doors_all) - set([first_choose,door_open]))[0]
    result_if_change_door.append(targets[second_choose])
    
    
fig, ax = plt.subplots()   
xdata1, ydata1 = [], []      
xdata2, ydata2 = [], []      
ln1, = ax.plot([], [], 'ro', animated=False,label="坚持第一次选择")  
ln2, = ax.plot([],[],'bx',animated=False,label="重新选择")

def init():
    ax.set_xlim(0,N_ROUND)
    ax.set_ylim(0,1)
    ax.set_xlabel("实验次数")
    ax.set_ylabel("获得car的概率")
    ax.set_yticks([k/10.0 for k in range(10)])
    return ln1,ln2

def update(n):
    if len(xdata1) > 0 and n == 0:
        return ln1,ln2
    x = n
    y = sum([is_car(x) for x in result_by_first_choose[0:n+1]]) / (n+1)
    xdata1.append(x)
    ydata1.append(y)
    ln1.set_data(xdata1,ydata1)
    y = sum([is_car(x) for x in result_if_change_door[0:n+1]]) / (n+1)
    xdata2.append(x)
    ydata2.append(y)
    print(x,y)
    ln2.set_data(xdata2,ydata2)
    return ln1,ln2
    
_animation  = animation.FuncAnimation(fig,update,frames=[k for k in range(0,N_ROUND,10)], init_func = init, blit=True)

plt.legend()
plt.title("三门问题模拟")
plt.show()  

# try:
#     writer = animation.writers['avconv']
# except KeyError:
#     writer = animation.writers['ffmpeg']
# writer = writer(fps=3)
# _animation.save("video.wav", writer=writer)
# plt.close(fig) 

