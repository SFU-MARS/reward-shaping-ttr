import gym
import gym_foo
from gym import wrappers
from time import *
from brs_engine.DubinsCar_brs_engine import *
from brs_engine.PlanarQuad_brs_engine import *

import ppo
import deepq
from utils.plotting_performance import *

import argparse
from utils.utils import *
import json
import pickle

from ppo1_utils.mlp_policy import MlpPolicy
from ppo1_utils.pposgd_simple import *
from collections import defaultdict
from baselines.common import tf_util as U
import numpy as np
from collections import Counter

from time import *

parser = argparse.ArgumentParser()
parser.add_argument("--load_path", help="where is the policy you want to evaluate.", type=str)
parser.add_argument("--load_iter", help="which iter of the policy you want to evaluate.", type=int)
# Two kinds of eval feedback. One is sparse_reward in the criterion of RL; another is min_cost in the criterion of optimal control
parser.add_argument("--feedback_type", help="which type of feedback to use.", type=str, default='hand_craft')
args = parser.parse_args()

# RUN_DIR = FIGURE_DIR = RESULT_DIR = None
RUN_DIR = args.load_path + '_iter_' + str(args.load_iter) + '_eval_record_add'
FIGURE_DIR = os.path.join(RUN_DIR, 'figure')
RESULT_DIR = os.path.join(RUN_DIR, 'result')

model_path = args.load_path + '/model'

gym_env = (args.load_path.split('/')[-1]).split('_')[0]
algo = None
if gym_env in ['DubinsCarEnv-v0', 'PlanarQuadEnv-v0']:
    algo = ppo
else:
    algo = deepq
print("env name:", gym_env)
env = gym.make(gym_env)
env.reward_type = args.feedback_type
env.set_hover_end = 'false'

if algo == ppo:

    assert gym_env == "DubinsCarEnv-v0" or gym_env == "PlanarQuadEnv-v0"

    maybe_mkdir(RUN_DIR)
    maybe_mkdir(FIGURE_DIR)
    maybe_mkdir(RESULT_DIR)

    num_iters = 10
    num_ppo_iters = 30
    timesteps_per_actorbatch = 128
    max_iters = num_ppo_iters

    # Initialize policy
    ppo.create_session()
    init_policy = ppo.create_policy('pi', env)
    ppo.initialize()

    pi = init_policy
    pi.load_model(model_path, iteration=args.load_iter)

    ppo_reward = list()
    ppo_length = list()
    suc_percents = list()
    wall_clock_time = list()

    # index for num_iters loop
    i = 1
    while i <= num_iters:
        wall_clock_time.append(time())
        print('overall training iteration %d' % i)
        pi, ep_mean_length, ep_mean_reward, suc_percent = algo.ppo_eval(env, pi, timesteps_per_actorbatch, max_iters, stochastic=False)

        ppo_length.extend(ep_mean_length)
        ppo_reward.extend(ep_mean_reward)
        suc_percents.append(suc_percent)

        plot_performance(range(len(ppo_reward)), ppo_reward, ylabel=r'avg reward per ppo-learning step',
                         xlabel='ppo iteration', figfile=os.path.join(FIGURE_DIR, 'ppo_reward'))
        plot_performance(range(len(suc_percents)), suc_percents,
                         ylabel=r'overall success percentage per algorithm step',
                         xlabel='algorithm iteration', figfile=os.path.join(FIGURE_DIR, 'success_percent'))

        # save data which is accumulated UNTIL iter i
        with open(RESULT_DIR + '/ppo_length_' + 'iter_' + str(i) + '.pickle', 'wb') as f1:
            pickle.dump(ppo_length, f1)
        with open(RESULT_DIR + '/ppo_reward_' + 'iter_' + str(i) + '.pickle', 'wb') as f2:
            pickle.dump(ppo_reward, f2)
        with open(RESULT_DIR + '/success_percent_' + 'iter_' + str(i) + '.pickle', 'wb') as fs:
            pickle.dump(suc_percents, fs)
        with open(RESULT_DIR + '/wall_clock_time_' + 'iter_' + str(i) + '.pickle', 'wb') as ft:
            pickle.dump(wall_clock_time, ft)

        # Incrementing our algorithm's loop counter
        i += 1

        # plot_performance(range(len(overall_perf)), overall_perf, ylabel=r'overall performance per algorithm step',
        #                  xlabel='algorithm iteration',
        #                  figfile=os.path.join(FIGURE_DIR, 'overall_perf'))

    env.close()

elif algo == deepq:
    pass

# Not sure how to proceed. Discussion tomorrow and see.