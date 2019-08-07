#!/usr/bin/env bash

BASEDIR=$(dirname "$0")
# echo "$BASEDIR"

# ------- training agent using TRPO algorithm -------
#/local-scratch/xlv/miniconda3/envs/py35_no_specific/bin/python3.5 $BASEDIR/quick_trpo.py --gym_env=PlanarQuadEnv-v0 --reward_type=ttr --set_additional_goal=angle
#/local-scratch/xlv/miniconda3/envs/py35_no_specific/bin/python3.5 $BASEDIR/quick_trpo.py --gym_env=PlanarQuadEnv-v0 --reward_type=hand_craft --set_additional_goal=angle
#/local-scratch/xlv/miniconda3/envs/py35_no_specific/bin/python3.5 $BASEDIR/quick_trpo.py --gym_env=PlanarQuadEnv-v0 --reward_type=distance --set_additional_goal=angle
/local-scratch/xlv/miniconda3/envs/py35_no_specific/bin/python3.5 $BASEDIR/quick_trpo.py --gym_env=PlanarQuadEnv-v0 --reward_type=distance_lambda_0.1 --set_additional_goal=angle
/local-scratch/xlv/miniconda3/envs/py35_no_specific/bin/python3.5 $BASEDIR/quick_trpo.py --gym_env=PlanarQuadEnv-v0 --reward_type=distance_lambda_1 --set_additional_goal=angle
/local-scratch/xlv/miniconda3/envs/py35_no_specific/bin/python3.5 $BASEDIR/quick_trpo.py --gym_env=PlanarQuadEnv-v0 --reward_type=distance_lambda_10 --set_additional_goal=angle
# python3.5 $BASEDIR/quick_trpo.py --gym_env=DubinsCarEnv-v0 --reward_type=ttr

# run 5 times for each reward type using trpo algorithm for quadrotor task
#for VARIABLE in 1 2 3 4 5
#do
#    python3.5 $BASEDIR/train_trpo.py --gym_env=PlanarQuadEnv-v0 --reward_type=ttr --algo=trpo --set_angle_goal=false
#    python3.5 $BASEDIR/train_trpo.py --gym_env=PlanarQuadEnv-v0 --reward_type=hand_craft --algo=trpo --set_angle_goal=false
#    python3.5 $BASEDIR/train_trpo.py --gym_env=PlanarQuadEnv-v0 --reward_type=distance --algo=trpo --set_angle_goal=false
#done
# ---------------------------------------------------




# ------- training agent using PPO algorithm -------
# python3.5 $BASEDIR/train_ppo.py --gym_env=PlanarQuadEnv-v0 --reward_type=ttr --algo=ppo --set_additional_goal=angle
#python3.5 $BASEDIR/train_ppo.py --gym_env=PlanarQuadEnv-v0 --reward_type=distance --algo=ppo --set_additional_goal=angle
#python3.5 $BASEDIR/train_ppo.py --gym_env=PlanarQuadEnv-v0 --reward_type=hand_craft --algo=ppo --set_additional_goal=angle
#python3.5 $BASEDIR/train_ppo.py --gym_env=PlanarQuadEnv-v0 --reward_type=distance_lambda_0.1 --algo=ppo --set_additional_goal=angle
#python3.5 $BASEDIR/train_ppo.py --gym_env=PlanarQuadEnv-v0 --reward_type=distance_lambda_1 --algo=ppo --set_additional_goal=angle
#python3.5 $BASEDIR/train_ppo.py --gym_env=PlanarQuadEnv-v0 --reward_type=distance_lambda_10 --algo=ppo --set_additional_goal=angle

# python3.5 $BASEDIR/train_ppo.py --gym_env=DubinsCarEnv-v0 --reward_type=ttr --algo=ppo

# run 5 times for each reward type using ppo algorithm for quadrotor task
#for VARIABLE in 1 2 3 4 5
#do
#    python3.5 $BASEDIR/train_ppo.py --gym_env=PlanarQuadEnv-v0 --reward_type=ttr --algo=ppo --set_angle_goal=false
#    python3.5 $BASEDIR/train_ppo.py --gym_env=PlanarQuadEnv-v0 --reward_type=hand_craft --algo=ppo --set_angle_goal=false
#    python3.5 $BASEDIR/train_ppo.py --gym_env=PlanarQuadEnv-v0 --reward_type=distance --algo=ppo --set_angle_goal=false
#done
# --------------------------------------------------

# ------- training agent using DDPG algorithm -------
#/local-scratch/xlv/miniconda3/envs/py35_no_specific/bin/python3.5 $BASEDIR/train_ddpg.py --env_id=PlanarQuadEnv-v0 --reward_type=ttr --set_additional_goal=angle
#python3.5 $BASEDIR/train_ddpg.py --env_id=PlanarQuadEnv-v0 --reward_type=distance --set_additional_goal=angle
#python3.5 $BASEDIR/train_ddpg.py --env_id=PlanarQuadEnv-v0 --reward_type=hand_craft --set_additional_goal=angle
#python3.5 $BASEDIR/train_ddpg.py --env_id=PlanarQuadEnv-v0 --reward_type=distance_lambda_0.1 --set_additional_goal=angle
#python3.5 $BASEDIR/train_ddpg.py --env_id=PlanarQuadEnv-v0 --reward_type=distance_lambda_1 --set_additional_goal=angle
#python3.5 $BASEDIR/train_ddpg.py --env_id=PlanarQuadEnv-v0 --reward_type=distance_lambda_10 --set_additional_goal=angle

# python3.5 $BASEDIR/train_ddpg.py --env-id=DubinsCarEnv-v0 --reward_type=ttr
# ---------------------------------------------------



#python3.5 $BASEDIR/train_ppo.py --gym_env=PlanarQuadEnv-v0 --reward_type=distance_lambda_10 --algo=ppo --set_additional_goal=angle
#python3.5 $BASEDIR/train_ddpg.py --env_id=PlanarQuadEnv-v0 --reward_type=distance_lambda_10 --set_additional_goal=angle
#python3.5 $BASEDIR/train_ppo.py --gym_env=PlanarQuadEnv-v0 --reward_type=distance_lambda_1 --algo=ppo --set_additional_goal=angle
#python3.5 $BASEDIR/train_ddpg.py --env_id=PlanarQuadEnv-v0 --reward_type=distance_lambda_1 --set_additional_goal=angle
#python3.5 $BASEDIR/train_ppo.py --gym_env=PlanarQuadEnv-v0 --reward_type=distance_lambda_0.1 --algo=ppo --set_additional_goal=angle
#python3.5 $BASEDIR/train_ddpg.py --env_id=PlanarQuadEnv-v0 --reward_type=distance_lambda_0.1 --set_additional_goal=angle