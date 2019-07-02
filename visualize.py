# 画图用到的函数
# liyi，2019/7/1

import numpy as np

def smooth(step,value,window=10000,overlap=0.2):
    # 对 episode value 进行加窗平滑 7/1
    # 返回窗内的均值和标准差
    smooth_step = []
    smooth_value_mean = []
    smooth_value_std = []
    down_limit = 0
    up_limit = window
    count = 0
    while up_limit<=step[-1]:
        smooth_step += [count]
        temp_value = value[np.array(step>=down_limit) & np.array(step<up_limit)]
        smooth_value_mean += [temp_value.mean()]
        smooth_value_std += [temp_value.std()]
        count += 1
        down_limit += window*overlap
        up_limit += window*overlap
    #print(len(smooth_step))
    #print(smooth_value_mean)
    #print(smooth_value_std)
    return np.array(smooth_step), np.array(smooth_value_mean), np.array(smooth_value_std)
    
def smooth2(reward, window=10):
    down_limit = 0
    up_limit = window
    smooth_reward = []
    while up_limit <= len(reward):
        temp_reward = reward[down_limit:up_limit]
        smooth_reward += [temp_reward.mean()]
        down_limit += window
        up_limit += window
        
    return smooth_reward
    
    
def statistic(value_list):
    # 输入的是一个 list，每个元素都是一组平滑处理后的 episode value 实验数据
    # 返回多个实验的均值和方差
    lengthes = [len(v) for v in value_list]
    length = min(lengthes)
    value_list = [v[0:length] for v in value_list]
    value = np.stack(value_list)
    value_mean = value.mean(axis=0)
    value_std = value.std(axis=0)
    return value_mean, value_std
    
    