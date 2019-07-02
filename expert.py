# Implementation of https://arxiv.org/abs/1801.10459
# Pretraining Deep Actor-Critic Reinforcement Learning Algorithms With Expert Demonstrations
# Xiaoqin Zhang, Huimin Ma
# liyi, 2019/6/27

import os
import random
import torch
import numpy as np
import h5py
from torch.utils.data import Dataset


class Expert(Dataset):
    def __init__(self, expert_dir, buffer_size=1e6):
        expert_files = sorted([os.path.join(expert_dir, f)
                        for f in os.listdir(expert_dir) if '.pt' in f])  ### 读取目录下所有 pt 文件 6/27
        temp_trajectories = torch.load(expert_files[0])
        num_samples_per_file = torch.prod(torch.tensor(temp_trajectories['states'].shape[0:2]))###一个文件中包含多少expert timestep6/28
        num_files = np.minimum(len(expert_files), int(buffer_size/num_samples_per_file))   ### 随机取出来的文件个数 6/28
        start_idx = np.random.randint(len(expert_files)-num_files)
        sample_files = expert_files[start_idx:start_idx+num_files]  ### 为了计算 expert value，不能打乱 6/28
        
        self.trajectories = {}
        num_trajectories = num_files * num_samples_per_file
        for sample_file in sample_files:
            temp_trajectories = torch.load(sample_file)
        
            for k,v in temp_trajectories.items():
                if k not in self.trajectories.keys():
                    self.trajectories[k] = [v]
                else:
                    self.trajectories[k] += [v]
                    
        for k,v in self.trajectories.items():
            self.trajectories[k] = torch.cat(self.trajectories[k])

        self.i2traj_idx = {}
        self.i2i = {}
        
        self.length = self.trajectories['lengths'].sum().item()

        traj_idx = 0
        i = 0

        self.get_idx = []
        
        for j in range(self.length):
            
            while self.trajectories['lengths'][traj_idx].item() <= i:
                i -= self.trajectories['lengths'][traj_idx].item()
                traj_idx += 1

            self.get_idx.append((traj_idx, i))   ### 把 j 映射为元组二维坐标 (traj_idx, i) 6/27

            i += 1

        #print(self.get_idx[0:100])   ### 6/27
        #print(self.get_idx[200:300])
        print('Expert dataset initialized!')
        print('length ',self.length)
        print('states ', self.trajectories['states'].shape)
            
            
    def __len__(self):
        return self.length

    def __getitem__(self, i):
        traj_idx, i = self.get_idx[i]

        return self.trajectories['states'][traj_idx][i], self.trajectories['actions'][traj_idx][i], \
            self.trajectories['rewards'][traj_idx][i], self.trajectories['done'][traj_idx][i]
        #return self.trajectories['states'][traj_idx][i], self.trajectories['actions'][traj_idx][i]
    
    def value(self, num_episodes=10, gamma=0.99):
        ### 计算 expert 的 value
        idx = np.random.randint(int(self.length/10))
        trajectory = self.__getitem__(idx)
        while not trajectory[3] and idx < self.length:   ### 找到第一个 episode 的起点 6/28
            idx += 1
            trajectory = self.__getitem__(idx)
            
        idx += 1
        expert_value = 0.
        for _ in range(num_episodes):
            temp_value = 0.
            discount = 1
            trajectory = self.__getitem__(idx)
            temp_value += discount * trajectory[2]
            discount *= gamma
            #a = idx
            #print('idx', idx)
            while not trajectory[3]:
                idx += 1
                trajectory = self.__getitem__(idx)
                temp_value += discount * trajectory[2]
                #print('episode ', idx, trajectory[2])
                discount *= gamma
            idx += 1
            #print('length', idx - a)
            expert_value += temp_value
            
        expert_value /= num_episodes
        return expert_value
        

    def sample(self, batch_size):   ### 类似 replay_buffer 的作用？ 6/27
        ind = np.random.randint(0, self.length, size=batch_size)
        
        x, y = [], []

        for i in ind: 
            X, Y, _, _ = self.__getitem__(i)
            #X, Y = self.__getitem__(i)
            x.append(np.array(X, copy=True))
            y.append(np.array(Y, copy=True))

        return np.array(x), np.array(y)
    
    
    