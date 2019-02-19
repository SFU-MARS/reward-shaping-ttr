#!/bin/sh
# This is a comment!
source ~/.bashrc
killgazebogym
cd launch
roslaunch QuadrotorAirSpace_v0.launch
python3 train.py --gym_env=PlanarQuadEnv-v0 --reward_type=ttr --algo=ppo
python3 train.py --gym_env=PlanarQuadEnv-v0 --reward_type=hand_craft --algo=ppo
