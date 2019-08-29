import os,sys
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

import pickle
from baselines import logger
import ppo
import gym
from train_ppo import *

import globals
from time import *
from utils.plotting_performance import *
from utils.utils import *

from brs_engine.DubinsCar_brs_engine import *
from brs_engine.PlanarQuad_brs_engine import *



# ranges: [(1,10), (20,30), (40,50), ..]
def gen_plots(args, ranges):
    if args['loadpath'] == None:
        raise ValueError("Loading path is None!!")
    if args['gym_env'] == "PlanarQuadEnv-v0":
        lo = [-5, 0, -np.pi]
        hi = [5, 10, np.pi]
        bins = [100, 100, 100]
    elif args['gym_env'] == "DubinsCarEnv-v0":
        lo = [-5, -5, 0]
        hi = [5, 5, np.pi * 2]
        bins = [100, 100, 100]
    else:
        raise ValueError("env's name is invalid!!")

    full_dir = args['loadpath']+'/heatmap'
    num_files = len([x for x in os.listdir(full_dir)])
    # print("num files:", num_files)

    # fig = plt.figure()

    num_fig = len(ranges)
    fig, axes = plt.subplots(2, num_fig)
    print(axes)
    for rg in ranges:
        if len(rg) != 2 or rg[0] > rg[1] or rg[0] > num_files or rg[1] > num_files:
            raise ValueError("Please provide valid range series!!")
        else:
            rg_lo = rg[0]
            rg_hi = rg[1]
            all_obs_cur_rg = []
            all_dones_cur_rg = []
            for id in range(rg_lo, rg_hi):
                with open(full_dir + '/iter_' + str(id) + '.pkl', 'rb') as f:
                    seg_info = pickle.load(f)
                # obs: np.array([[_,_,_],
                #                [_,_,_]])
                all_obs_cur_rg.append(seg_info['ob'])
            all_obs_cur_rg_concat = np.concatenate(all_obs_cur_rg, axis=0)
            if args['gym_env'] == "PlanarQuadEnv-v0":
                x = all_obs_cur_rg_concat[:, 0]
                z = all_obs_cur_rg_concat[:, 2]
                theta = all_obs_cur_rg_concat[:,4]
                # ax = fig.add_subplot(1,num_fig,1+ranges.index(rg), adjustable='box', aspect=1.0)

                # ax.set_xlabel("X position")
                # ax.set_ylabel("Z position")
                # ax.hist2d(x, z, bins=min(bins[:2]), range=[[-5, 5], [0, 10]])
                # ax.hist2d(x, z, bins=min(bins[:2]))

                # theta as 1d hist, xz as 2d hist
                # axes[0, ranges.index(rg)].set(adjustable='box', aspect='equal')
                # axes[0, ranges.index(rg)].hist(theta, bins=bins[2], density=True, facecolor='g', alpha=0.75, log=True)
                # axes[1, ranges.index(rg)].set(adjustable='box', aspect='equal')
                # axes[1, ranges.index(rg)].hist2d(x, z, range=[[-5,5],[0,10]], bins=min(bins[:2]), cmap='tab20')

                # x as 1d hist, z&theta as 2d hist
                # axes[0, ranges.index(rg)].set(adjustable='box', aspect='equal')
                # axes[0, ranges.index(rg)].hist(x, bins=bins[0], density=True, facecolor='g', alpha=0.75, log=True)
                # axes[1, ranges.index(rg)].set(adjustable='box', aspect='equal')
                # axes[1, ranges.index(rg)].hist2d(z,theta, range=[[0,10],[-np.pi,np.pi]], bins=min(bins[1:]), cmap='tab20')

                # z as 1d hist, x&theta as 2d hist
                axes[0, ranges.index(rg)].set(adjustable='box', aspect='equal')
                axes[0, ranges.index(rg)].hist(z, bins=bins[1], density=True, facecolor='g', alpha=0.75, log=True)
                axes[1, ranges.index(rg)].set(adjustable='box', aspect='equal')
                axes[1, ranges.index(rg)].hist2d(x, theta, range=[[-5, 5], [-np.pi, np.pi]], bins=bins[0],
                                                 cmap='tab20')
            elif args['gym_env'] == "DubinsCarEnv-v0":
                xy = all_obs_cur_rg_concat[:, :2]
                ax = fig.add_subplot(1,num_fig,1+ranges.index(rg))
                ax.set_xlabel('X position')
                ax.set_ylabel('Y position')
                ax.hist2d(xy[:, 0], xy[:, 1], bins=min(bins[:2]))
            else:
                ValueError("env's name is invalid!!")
    # for ax in axes:
    #     ax.set_xlabel('Common x-label')
    #     ax.set_ylabel('Common y-label')
    # fig.add_subplot(111, frameon=False)
    # plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
    # plt.xlabel('Common x-label')
    # plt.ylabel('Common z-label')
    fig.tight_layout()
    plt.show()

    plt.savefig(args['loadpath'] + '/test2_xtheta.png')
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gym_env", help="which gym environment to use.", type=str, default='PlanarQuadEnv-v0')
    parser.add_argument("--loadpath", help="the path you load for generating heatmaps", type=str, default=None)
    args = parser.parse_args()
    args = vars(args)

    # gen_plots(args, [(1,20), (20,40), (40,60), (60,80), (80,100), (100,120)])
    gen_plots(args, [(1,20), (20,40), (80,100)])





# -----------  this part below is for generating obs during training for building heatmaps later ---------------------
# globals.init()
# def gen_iter_obs(args):
#     assert args['gym_env'] == 'DubinsCarEnv-v0' or args['gym_env'] == 'PlanarQuadEnv-v0'
#     logger.configure(dir=args['RUN_DIR'])
#     globals.g_hm_dirpath = args['HEATMAP_DIR']
#     globals.g_iter_id = 0
#
#     env = gym.make(args['gym_env'])
#     env.reward_type = args['reward_type']
#     env.set_additional_goal = args['set_additional_goal']
#     logger.record_tabular("algo", args['algo'])
#     logger.record_tabular("env", args['gym_env'])
#     logger.record_tabular("env.set_additional_goal", args['set_additional_goal'])
#     logger.record_tabular("env.reward_type", args['reward_type'])
#
#     logger.dump_tabular()
#
#
#     # Initialize brs engine. You also have to call reset_variables() after instance initialization
#     if args['reward_type'] == 'ttr':
#         if args['gym_env'] == 'DubinsCarEnv-v0':
#             brsEngine = DubinsCar_brs_engine()
#             brsEngine.reset_variables()
#         elif args['gym_env'] == 'PlanarQuadEnv-v0':
#             brsEngine = Quadrotor_brs_engine()
#             brsEngine.reset_variables()
#         else:
#             raise ValueError("invalid environment name for ttr reward!")
#         # You have to assign the engine
#         env.brsEngine = brsEngine
#     elif args['reward_type']  in ['hand_craft', 'distance', 'distance_lambda_10', 'distance_lambda_1',
#                               'distance_lambda_0.1']:
#         pass
#     else:
#         raise ValueError("wrong type of reward")
#     ppo_params_json = os.environ['PROJ_HOME'] + '/ppo_params.json'
#
#     # Start to train the policy
#     trained_policy = train(env=env, algorithm=ppo, params=ppo_params_json, args=args, save_obs=True)
#     trained_policy.save_model(args['MODEL_DIR'])
#
#     return True
#
# if __name__ == "__main__":
#     with tf.device('/gpu:1'):
#         # ----- path setting ------
#         parser = argparse.ArgumentParser()
#         parser.add_argument("--gym_env", help="which gym environment to use.", type=str, default='PlanarQuadEnv-v0')
#         parser.add_argument("--reward_type", help="which type of reward to use.", type=str, default='ttr')
#         parser.add_argument("--algo", help="which type of algorithm to use.", type=str, default='ppo')
#         parser.add_argument("--set_additional_goal", type=str, default="angle")
#         args = parser.parse_args()
#         args = vars(args)
#
#         RUN_DIR = os.path.join(os.getcwd(), 'heatmaps_icra',
#                                args['gym_env']+ '_' + args['reward_type'] + '_' + args['algo'] + '_' + strftime(
#                                    '%d-%b-%Y_%H-%M-%S'))
#         MODEL_DIR = os.path.join(RUN_DIR, 'model')
#         FIGURE_DIR = os.path.join(RUN_DIR, 'figure')
#         RESULT_DIR = os.path.join(RUN_DIR, 'result')
#         HEATMAP_DIR = os.path.join(RUN_DIR, 'heatmap')
#
#         maybe_mkdir(RUN_DIR)
#         maybe_mkdir(MODEL_DIR)
#         maybe_mkdir(FIGURE_DIR)
#         maybe_mkdir(RESULT_DIR)
#         maybe_mkdir(HEATMAP_DIR)
#
#         args['RUN_DIR'] = RUN_DIR
#         args['MODEL_DIR'] = MODEL_DIR
#         args['FIGURE_DIR'] = FIGURE_DIR
#         args['RESULT_DIR'] = RESULT_DIR
#         args['HEATMAP_DIR'] = HEATMAP_DIR
#         gen_iter_obs(args)