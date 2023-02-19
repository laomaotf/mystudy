######################################
#参考 https://zhuanlan.zhihu.com/p/549588865?utm_campaign=&utm_medium=social&utm_oi=1219637155305357312&utm_psn=1597142465382981632&utm_source=com.yuque.mobile.android.app
#An Efficient Algorithm for Automatic Peak Detection in Noisy Periodic and Quasi-Periodic Signals

#针对周期或准周期信号，给定样本包含多个周期，确保目标信号的峰值出现次数比噪声峰值出现次数多
######################################

import numpy as np

import matplotlib.pyplot as plt

def AMPD(data,topK=0):
    """
    实现AMPD算法
    :param data: 1-D numpy.ndarray 
    :return: 波峰所在索引值的列表
    """
    p_data = np.zeros_like(data, dtype=np.int32)
    count = data.shape[0]
    arr_rowsum = []
    for k in range(1, count // 2 + 1): 
        #搜索一个scale，该scale对应的peak点最多，这个scale就是目标信号的周期T
        row_sum = 0
        for i in range(k, count - k):
            if data[i] > data[i - k] and data[i] > data[i + k]:
                row_sum -= 1
        arr_rowsum.append(row_sum)
    min_index = np.argmin(arr_rowsum)
    max_window_length = min_index
    for k in range(1, max_window_length + 1): #
        for i in range(k, count - k):
            if data[i] > data[i - k] and data[i] > data[i + k]:
                p_data[i] += 1
    print(max_window_length)
    return np.where(p_data == max_window_length-topK)[0]



def sim_data():
    N = 100
    x = np.linspace(0, 200, N)
    y = 2 * np.cos(2 * np.pi * 300 * x) \
        + 5 * np.sin(2 * np.pi * 100 * x)  \
        + 4 * np.random.randn(N)
    
    y = 2 * np.cos(2 * np.pi * 5 * x) 
    return y

def vis():
    y = sim_data()
    plt.plot(range(len(y)), y)
    px = AMPD(y)
    plt.scatter(px, y[px], color="red")

    plt.show()

vis()