#!/local-scratch/xlv/miniconda3/envs/py35_no_specific/bin/python

import numpy as np
import os, sys
import gym
from gym_foo import gym_foo

# import spinup

import trpo_mpi


if __name__ == "__main__":
    #import argparse

    #parser = argparse.ArgumentParser()
    #parser.add_argument('--gym_env', type=str, default='PlanarQuadEnv-v0')
    #parser.add_argument('--reward_type', type=str, default='sparse')
    #parser.add_argument('--algo', type=str, default='ppo')
    #parser.add_argument('--set_hover_end', type=str, default='false')
    #args = parser.parse_args()

    #kwargs = {'reward_type':args.reward_type, 'set_hover_end':args.set_hover_end}
    #print(kwargs)
    #spinup.ppo(lambda:gym.make(args.gym_env, **kwargs))
    

    # TRPO optimizing for current task
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--gym_env', type=str, default='PlanarQuadEnv-v0')
    parser.add_argument('--reward_type', type=str, default='hand_craft')
    parser.add_argument('--algo', type=str, default='trpo')
    parser.add_argument('--set_angle_goal', type=str, default='false')
    args = parser.parse_args()

    kwargs = {'reward_type':args.reward_type, 'set_angle_goal':args.set_hover_end}
    env = gym.make(args.gym_env)
    env.reward_type= args.reward_type
    env.set_angle_goal= args.set_angle_goal

    trpo_mpi.learn(network='mlp', env=env, total_timesteps=1e6)
