# Implementation of https://arxiv.org/abs/1801.10459
# Pretraining Deep Actor-Critic Reinforcement Learning Algorithms With Expert Demonstrations
# Xiaoqin Zhang, Huimin Ma
# liyi, 2019/6/27

import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import utils

from DDPG import Actor, Critic
from expert import Expert

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class ExpertDDPG(object):
    def __init__(self, state_dim, action_dim, max_action):
        self.actor = Actor(state_dim, action_dim, max_action).to(device)
        self.actor_target = Actor(state_dim, action_dim, max_action).to(device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=1e-4)

        self.critic = Critic(state_dim, action_dim).to(device)
        self.critic_target = Critic(state_dim, action_dim).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), weight_decay=1e-2)        

    def select_action(self, state):
        state = torch.FloatTensor(state.reshape(1, -1)).to(device)
        return self.actor(state).cpu().data.numpy().flatten()

    def value(self, state):
        ### 计算 Q(s, pi(s)) 6/28
        state = torch.FloatTensor(state.reshape(1, -1)).to(device)
        return self.critic(state, self.actor(state))

    def train(self, expert, replay_buffer, iterations, batch_size=64, discount=0.99,
              tau=0.001, lambda_Q = 1, lambda_pi = 1, use_expert=True):

        for it in range(iterations):

            ### expert demostration 6/27
            #expert = Expert(expert_dir = './expert_data/')
            state_star, action_star = expert.sample(batch_size)
            state_star = torch.FloatTensor(state_star).to(device)
            action_star = torch.FloatTensor(action_star).to(device)
            
            # Sample replay buffer 
            x, y, u, r, d = replay_buffer.sample(batch_size)
            state = torch.FloatTensor(x).to(device)
            action = torch.FloatTensor(u).to(device)
            next_state = torch.FloatTensor(y).to(device)
            done = torch.FloatTensor(1 - d).to(device)
            reward = torch.FloatTensor(r).to(device)

            # Compute the target Q value
            target_Q = self.critic_target(next_state, self.actor_target(next_state))
            target_Q = reward + (done * discount * target_Q).detach()

            # Get current Q estimate
            current_Q = self.critic(state, action)

            ### 计算 A(s*, a*) 6/27
            A = self.critic(state_star, action_star) - self.critic(state_star, self.actor(state_star))
            EA = torch.mean(A, 0)    ### 求期望，忽略 gamma 6/28
            
            # Compute critic loss
            critic_loss = F.mse_loss(current_Q, target_Q)
            #print(critic_loss)
            #print(lambda_Q * torch.max(input=EA, other=torch.tensor(0.).to(device)))
            if use_expert:
                critic_loss += lambda_Q * torch.max(input=-EA, other=torch.tensor(0.).to(device)).item() ### ours 6/28
            
            # Optimize the critic
            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            self.critic_optimizer.step()

            # Compute actor loss
            actor_loss = -self.critic(state, self.actor(state)).mean()
            if use_expert:
                actor_loss += lambda_pi * EA.item()     ### ours 6/28
            
            # Optimize the actor 
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            # Update the frozen target models
            for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

            for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)


    def save(self, filename, directory):
        torch.save(self.actor.state_dict(), '%s/%s_actor.pth' % (directory, filename))
        torch.save(self.critic.state_dict(), '%s/%s_critic.pth' % (directory, filename))


    def load(self, filename, directory):
        self.actor.load_state_dict(torch.load('%s/%s_actor.pth' % (directory, filename)))
        self.critic.load_state_dict(torch.load('%s/%s_critic.pth' % (directory, filename)))