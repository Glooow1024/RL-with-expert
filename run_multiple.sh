#!/bin/bash

# Script to reproduce results

for ((i=0;i<5;i+=1))
do 
    python Oursmain.py --policy_name ExpertDDPG --env_name Hopper-v2 --save_models --seed $i --use_expert
    
    python Oursmain.py --policy_name ExpertDDPG --env_name Hopper-v2 --save_models --seed $i
    
    python Oursmain.py --policy_name ExpertDDPG --env_name Walker2d-v2 --save_models --seed $i --use_expert
    
    python Oursmain.py --policy_name ExpertDDPG --env_name Walker2d-v2 --save_models --seed $i

done
