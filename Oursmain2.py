import numpy as np
import torch
import gym
import argparse
import os

import utils
import TD3
import OurDDPG
import DDPG
import ExpertDDPG
from expert import Expert


# Runs policy for X episodes and returns average reward
def evaluate_policy(policy, eval_episodes=10):
    avg_reward = 0.    ### ground truth 6/28
    our_reward = 0.    ### critic estimation 6/28
    our_reward_contrast = 0.
    gamma = 0.99
    for _ in range(eval_episodes):
        obs = env.reset()
        our_reward += policy.value(obs)
        if policy_contrast is not None:
            our_reward_contrast += policy_contrast.value(obs)
        done = False
        discount = 1
        while not done:
            action = policy.select_action(np.array(obs))
            obs, reward, done, _ = env.step(action)
            avg_reward += discount * reward    ### 没有加 discount？ 6/28
            discount *= gamma

    avg_reward /= eval_episodes
    our_reward /= eval_episodes
    our_reward_contrast /= eval_episodes

    print("---------------------------------------")
    print("Evaluation over %d episodes: %f(GD) %f(Ours) %f(Contrast)" %
          (eval_episodes, avg_reward, our_reward, our_reward_contrast))
    print("---------------------------------------")
    return avg_reward, our_reward, our_reward_contrast


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy_name", default="TD3")                    # Policy name
    parser.add_argument("--env_name", default="HalfCheetah-v2")            # OpenAI gym environment name ### v1改为v2
    parser.add_argument("--seed", default=0, type=int)                    # Sets Gym, PyTorch and Numpy seeds
    parser.add_argument("--start_timesteps", default=1e4, type=int)        # How many time steps purely random policy is run for
    parser.add_argument("--eval_freq", default=5e3, type=float)            # How often (time steps) we evaluate
    parser.add_argument("--max_timesteps", default=1e6, type=float)        # Max time steps to run environment for
    parser.add_argument("--save_models", action="store_true")            # Whether or not models are saved
    parser.add_argument("--expl_noise", default=0.1, type=float)        # Std of Gaussian exploration noise
    parser.add_argument("--batch_size", default=100, type=int)            # Batch size for both actor and critic
    parser.add_argument("--discount", default=0.99, type=float)            # Discount factor
    parser.add_argument("--tau", default=0.005, type=float)                # Target network update rate
    parser.add_argument("--policy_noise", default=0.2, type=float)        # Noise added to target policy during critic update
    parser.add_argument("--noise_clip", default=0.5, type=float)        # Range to clip target policy noise
    parser.add_argument("--policy_freq", default=2, type=int)            # Frequency of delayed policy updates
    parser.add_argument("--expert_timesteps", default = 3e4, type=int)    ### 使用 expert policy 的 steps 6/28
    #parser.add_argument("--use_expert", action = "store_true")         ### 是否使用 expert 训练 6/28
    args = parser.parse_args()

    file_name = "%s_%s_%s" % (args.policy_name, args.env_name, str(args.seed))
    file_name_contrast = "%s_%s_%s_contrast" % (args.policy_name, args.env_name, str(args.seed))
    print("---------------------------------------")
    print("Settings: %s" % (file_name))
    print("---------------------------------------")

    if not os.path.exists("./results"):
        os.makedirs("./results")
    if args.save_models and not os.path.exists("./pytorch_models"):
        os.makedirs("./pytorch_models")

    env = gym.make(args.env_name)
    env_contrast = gym.make(args.env_name)  ### 必须要两个环境 6/28

    # Set seeds
    env.seed(args.seed)
    env_contrast.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0] 
    max_action = float(env.action_space.high[0])

    # Initialize policy
    if args.policy_name == "TD3": policy = TD3.TD3(state_dim, action_dim, max_action)
    elif args.policy_name == "OurDDPG": policy = OurDDPG.DDPG(state_dim, action_dim, max_action)
    elif args.policy_name == "DDPG": policy = DDPG.DDPG(state_dim, action_dim, max_action)
    elif args.policy_name == "ExpertDDPG": 
        policy = ExpertDDPG.ExpertDDPG(state_dim, action_dim, max_action)
        policy_contrast = ExpertDDPG.ExpertDDPG(state_dim, action_dim, max_action)  ### 不使用 expert 作为对比 6/28

    replay_buffer = utils.ReplayBuffer()
    replay_buffer_contrast = utils.ReplayBuffer()   ### 不能使用同一个经验池 6/28
    
    ### expert 6/28
    expert_dir = './expert_data/'
    expert = Expert(expert_dir)

    total_timesteps = 0
    timesteps_since_eval = 0
    episode_num = 0
    done = True 
    done_contrast = True
    expert_flag = True    ### 决定当前是否使用 expert policy 6/28
    
    # Evaluate untrained policy
    evaluations = [(total_timesteps, evaluate_policy(policy, policy_contrast))]    ### tuple 6/28

    while total_timesteps < args.max_timesteps:
        
        '''################### without expert #####################
        if done_contrast: 

            if total_timesteps != 0: 
                print((" "*49 + "(Contrast)Reward: %f") % (episode_reward_contrast))
                if args.policy_name == "TD3":
                    pass
                else: 
                    policy_contrast.train(expert, replay_buffer_contrast, episode_timesteps_contrast,
                                 args.batch_size, args.discount, args.tau, use_expert=False)
            
            # Reset environment
            obs_contrast = env_contrast.reset()
            done_contrast = False
            episode_reward_contrast = 0
            episode_timesteps_contrast = 0
            
        # Select action randomly or according to policy
        ### 刚开始只是随机采样 action 6/28
        ### 到一定时间之后再用 policy 计算 action 6/28
        if total_timesteps < args.start_timesteps:
            action_contrast = env_contrast.action_space.sample()
        else:
            action_contrast = policy_contrast.select_action(np.array(obs_contrast))
            if args.expl_noise != 0: 
                action_contrast = (action_contrast + np.random.normal(0, args.expl_noise, size=env_contrast.action_space.shape[0])).clip(env_contrast.action_space.low, env_contrast.action_space.high)'''

        # Perform action
        ### 在仿真环境中执行 action 并观测 state 和 reward 6/28
        new_obs_contrast, reward_contrast, done_contrast, _ = env_contrast.step(action_contrast) 
        done_bool_contrast = 0 if episode_timesteps_contrast + 1 == env_contrast._max_episode_steps \
            else float(done_contrast)
        episode_reward_contrast += reward_contrast    ### 一个 episode 中 reward 是累加的 6/28

        # Store data in replay buffer
        replay_buffer_contrast.add((obs_contrast, new_obs_contrast,
                                    action_contrast, reward_contrast, done_bool_contrast))

        obs_contrast = new_obs_contrast
                       
        episode_timesteps_contrast += 1

        
        ################### with expert #####################
        expert_flag = total_timesteps < args.expert_timesteps
        
        if done: 

            if total_timesteps != 0: 
                print(("Total T: %d Episode Num: %d Episode T: %d Reward: %f") % (total_timesteps, episode_num, episode_timesteps, episode_reward))
                if args.policy_name == "TD3":
                    policy.train(replay_buffer, episode_timesteps, args.batch_size, args.discount, args.tau, args.policy_noise, args.noise_clip, args.policy_freq)
                else: 
                    policy.train(expert, replay_buffer, episode_timesteps,
                                 args.batch_size, args.discount, args.tau, expert_flag)  ### 6/28
            
            # Evaluate episode
            if timesteps_since_eval >= args.eval_freq:
                timesteps_since_eval %= args.eval_freq
                evaluations.append((total_timesteps, evaluate_policy(policy, policy_contrast)))
                
                if args.save_models: 
                    policy.save(file_name, directory="./pytorch_models")
                    policy_contrast.save(file_name_contrast, directory="./pytorch_models")
                np.save("./results/%s" % (file_name), evaluations) 
            
            # Reset environment
            obs = env.reset()
            done = False
            episode_reward = 0
            episode_timesteps = 0
            episode_num += 1 
            
        # Select action randomly or according to policy
        ### 刚开始只是随机采样 action 6/28
        ### 到一定时间之后再用 policy 计算 action 6/28
        if total_timesteps < args.start_timesteps:
            action = env.action_space.sample()
        else:
            action = policy.select_action(np.array(obs))
            if args.expl_noise != 0: 
                action = (action + np.random.normal(0, args.expl_noise, size=env.action_space.shape[0])).clip(env.action_space.low, env.action_space.high)

        # Perform action
        ### 在仿真环境中执行 action 并观测 state 和 reward 6/28
        new_obs, reward, done, _ = env.step(action) 
        done_bool = 0 if episode_timesteps + 1 == env._max_episode_steps else float(done)
        episode_reward += reward    ### 一个 episode 中 reward 是累加的 6/28

        # Store data in replay buffer
        replay_buffer.add((obs, new_obs, action, reward, done_bool))

        obs = new_obs
        episode_timesteps += 1
        
        ##################### share ####################
        total_timesteps += 1
        timesteps_since_eval += 1
        
    # Final evaluation 
    evaluations.append((total_timesteps, evaluate_policy(policy, policy_contrast)))
    if args.save_models: 
        policy.save("%s" % (file_name), directory="./pytorch_models")
        policy_contrast.save("%s" % (file_name_contrast), directory="./pytorch_models")
    np.save("./results/%s" % (file_name), evaluations)  
