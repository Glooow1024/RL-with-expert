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
    def __init__(self, expert_dir, num_files=10, num_trajectories_per_file=100, subsample_frequency=1):
        expert_files = sorted([os.path.join(expert_dir, f)
                        for f in os.listdir(expert_dir) if '.pt' in f])  ### 读取目录下所有 pt 文件 6/27
        expert_samples = random.sample(expert_files, num_files)   ### 随机选取 num_files 个文件
        #start_idx = np.random.randint(len(expert_files)-num_files)
        #sample_files = expert_files
        
        self.trajectories = {}
        all_trajectories = {}
        num_trajectories = num_files * num_trajectories_per_file
        for sample_file in expert_samples:
            temp_trajectories = torch.load(sample_file)
        
            perm = torch.randperm(temp_trajectories['states'].size(0))
            idx = perm[:num_trajectories_per_file]

            for k,v in temp_trajectories.items():
                if k not in all_trajectories.keys():
                    all_trajectories[k] = [v[idx]]
                else:
                    all_trajectories[k] += [v[idx]]
                    
        for k,v in all_trajectories.items():
            all_trajectories[k] = torch.cat(all_trajectories[k])
            
        # See https://github.com/pytorch/pytorch/issues/14886
        # .long() for fixing bug in torch v0.4.1
        start_idx = torch.randint(
            0, subsample_frequency, size=(num_trajectories, )).long()
            
        for k, v in all_trajectories.items():
            if k != 'lengths':
                samples = []
                for i in range(num_trajectories):
                    samples.append(v[i, start_idx[i]::subsample_frequency])
                self.trajectories[k] = torch.stack(samples)
            else:
                self.trajectories[k] = v // subsample_frequency

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

        return self.trajectories['states'][traj_idx][i], self.trajectories['actions'][traj_idx][i]
    
    def value(self):
        ### 计算 expert 的 value
        ### 未完待续。。。 6/28
        pass
        

    def sample(self, batch_size):   ### 类似 replay_buffer 的作用？ 6/27
        ind = np.random.randint(0, self.length, size=batch_size)
        
        x, y = [], []

        for i in ind: 
            X, Y = self.__getitem__(i)
            x.append(np.array(X, copy=True))
            y.append(np.array(Y, copy=True))

        return np.array(x), np.array(y)
    
    
    