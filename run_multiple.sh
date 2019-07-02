#!/bin/bash

# Script to reproduce results

for ((i=0;i<5;i+=1))
do 
    python Oursmain.py --policy_name ExpertDDPG \
        --env_name Hopper-v2 --save_models \
        --seed $i --use_expert \
        --expert_dir ./expert_data/Hopper-v2/
    
    python Oursmain.py --policy_name ExpertDDPG \
        --env_name Hopper-v2 --save_models \
        --seed $i \
        --expert_dir ./expert_data/Hopper-v2/
    
    python Oursmain.py --policy_name ExpertDDPG \
        --env_name Walker2d-v2 --save_models \
        --seed $i --use_expert \
        --expert_dir ./expert_data/Walker2d-v2/
    
    python Oursmain.py --policy_name ExpertDDPG \
        --env_name Walker2d-v2 --save_models \
        --seed $i \
        --expert_dir ./expert_data/Walker2d-v2/
        
    python Oursmain.py --policy_name ExpertDDPG \
        --env_name HalfCheetah-v2 --save_models \
        --seed $i --use_expert \
        --expert_dir ./expert_data/HalfCheetah-v2/
    
    python Oursmain.py --policy_name ExpertDDPG \
        --env_name HalfCheetah-v2 --save_models \
        --seed $i \
        --expert_dir ./expert_data/HalfCheetah-v2/

done
