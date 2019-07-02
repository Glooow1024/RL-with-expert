#python Oursmain.py --policy_name ExpertDDPG --env_name HalfCheetah-v2 \
#    --save_models --seed 9 --use_expert \
#    --expert_dir ./expert_data/HalfCheetah-v2/

SEED=2

python Oursmain.py --policy_name ExpertDDPG \
    --env_name Hopper-v2 --save_models \
    --seed $SEED --use_expert \
    --expert_dir ./expert_data/ppo_1/Hopper-v2/

python Oursmain.py --policy_name ExpertDDPG \
    --env_name Hopper-v2 --save_models \
    --seed $SEED \
    --expert_dir ./expert_data/ppo_1/Hopper-v2/

python Oursmain.py --policy_name ExpertDDPG \
    --env_name Walker2d-v2 --save_models \
    --seed $SEED --use_expert \
    --expert_dir ./expert_data/ppo_1/Walker2d-v2/
    
python Oursmain.py --policy_name ExpertDDPG \
    --env_name Walker2d-v2 --save_models \
    --seed $SEED \
    --expert_dir ./expert_data/ppo_1/Walker2d-v2/
    
python Oursmain.py --policy_name ExpertDDPG \
    --env_name HalfCheetah-v2 --save_models \
    --seed $SEED --use_expert \
    --expert_dir ./expert_data/ppo_1/HalfCheetah-v2/
    
python Oursmain.py --policy_name ExpertDDPG \
    --env_name HalfCheetah-v2 --save_models \
    --seed $SEED \
    --expert_dir ./expert_data/ppo_1/HalfCheetah-v2/
