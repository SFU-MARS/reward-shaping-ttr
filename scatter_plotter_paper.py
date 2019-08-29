import os,sys
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import gym
from gym_foo import gym_foo
import ppo

import pickle
from utils.utils import *


import argparse
if __name__ == "__main__":

    np.random.seed(2019)

    parser = argparse.ArgumentParser()
    parser.add_argument("--gym_env", help="which gym environment to use.", type=str, default='DubinsCarEnv-v0')
    parser.add_argument("--reward_type", help="which type of reward to use.", type=str, default='hand_craft')
    parser.add_argument("--load_dir", type=str, default=None)
    parser.add_argument("--load_iter", type=int, default=1)

    args = parser.parse_args()


    TRAJ_DIR = os.path.join(os.getcwd(), 'trajs_icra',
                            (args.load_dir).split('/')[-2] + '_' + 'iter_' + str(args.load_iter))
    maybe_mkdir(TRAJ_DIR)



    # Initialize env
    env = gym.make(args.gym_env)
    env.reward_type = args.reward_type
    env.set_additional_goal = 'angle'

    # Initialize policy
    ppo.create_session()
    init_policy = ppo.create_policy('pi', env)
    ppo.initialize()

    pi = init_policy
    pi.load_model(args.load_dir, iteration=args.load_iter)


    _, _, eval_ep_mean_reward, eval_suc_percent, trajs, dones = ppo.ppo_eval(env, pi, timesteps_per_actorbatch = 128,
                                                                     max_iters=5, stochastic=False, scatter_collect=True)
    # save the traj data to pkl
    with open(TRAJ_DIR + '/trajs.pkl', 'wb') as f1:
        pickle.dump(trajs, f1)
    with open(TRAJ_DIR + '/dones.pkl', 'wb') as f2:
        pickle.dump(dones, f2)

    print(trajs)
    print(dones)

    trajs = np.concatenate(trajs, axis=0)

    print("total steps:", np.shape(trajs))
    fig = plt.figure()

    if args.gym_env == 'DubinsCarEnv-v0':
        xys = trajs[:, :2]
        ax = fig.add_subplot(111)
        ax.set_title("2d histagram for simple car task")
        ax.set_xlabel('X pos')
        ax.set_ylabel('Y pos')
        ax.hist2d(xys[:, 0], xys[:, 1], bins=100)


    elif args.gym_env == 'PlanarQuadEnv-v0':
        xs = trajs[:, 0]
        zs = trajs[:, 2]
        ts = trajs[:, 4]

        print("len of xs:", np.shape(xs))
        print("len of zs:", np.shape(zs))
        print("len of ts:", np.shape(ts))

        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(xs, zs, ts, marker="o", color="r")
        ax.set_title("3D scatter plot for quadrotor task")
        ax.set_xlabel("X pos")
        ax.set_ylabel("Z pos")
        ax.set_zlabel("Pitch angle")


    else:
        raise ValueError("invalid env name!!")

    plt.show()




