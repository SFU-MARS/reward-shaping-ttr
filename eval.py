import gym
import gym_foo
from gym import wrappers
from time import *
from brs_engine.DubinsCar_brs_engine import *
from brs_engine.PlanarQuad_brs_engine import *

import ppo
import deepq
import utils.liveplot as liveplot
from utils.plotting_performance import *

import argparse
from utils.utils import *
import json
import pickle

parser = argparse.ArgumentParser()
parser.add_argument("--load_path", help="where is the policy you want to evaluate.", type=str)
parser.add_argument("--load_iter", help="which iter of the policy you want to evaluate.", type=int)
# Two kinds of eval feedback. One is sparse_reward in the criterion of RL; another is min_cost in the criterion of optimal control
parser.add_argument("--feedback_type", help="which type of feedback to use.", type=str, default='sparse_reward')

args = parser.parse_args()

RUN_DIR = FIGURE_DIR = RESULT_DIR = None

RUN_DIR = args.load_path + 'iter_' + str(args.load_iter) + '_eval'
FIGURE_DIR = os.path.join(RUN_DIR, 'figure')
RESULT_DIR = os.path.join(RUN_DIR, 'result')

model_path = args.load_path + '/model'

gym_env = (args.load_path.split('/')[-1]).split('_')[0]
algo = None
if gym_env in ['DubinsCarEnv-v0', 'PlanarQuadEnv-v0']:
    algo = ppo
else:
    algo = deepq

if algo == ppo:
    assert args.gym_env == "DubinsCarEnv-v0" or args.gym_env == "PlanarQuadEnv-v0"
    # Initialize policy
    ppo.create_session()
    init_policy = ppo.create_policy('pi', gym_env)
    ppo.initialize()

    pi = init_policy
    pi.load_model(model_path, iteration=args.load_iter)
