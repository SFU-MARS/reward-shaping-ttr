
import gym
import gym_foo
from gym import wrappers
from time import strftime
from brs_engine.DubinsCar_brs_engine import *
from brs_engine.PlanarQuad_brs_engine import *

import ppo
import utils.liveplot as liveplot
from utils.plotting_performance import *

import argparse
from utils.utils import *
import json
import pickle

parser = argparse.ArgumentParser()
parser.add_argument("--gym_env", help="which gym environment to use.", type=str, default='DubinsCarEnv-v0')
parser.add_argument("--reward_type", help="which type of reward to use.", type=str, default='hand_craft')
args = parser.parse_args()

RUN_DIR = os.path.join(os.getcwd(), 'runs', args.gym_env + '_' + args.reward_type + '_' + strftime('%d-%b-%Y_%H-%M-%S'))
MODEL_DIR = os.path.join(RUN_DIR, 'model')
FIGURE_DIR = os.path.join(RUN_DIR, 'figure')
RESULT_DIR = os.path.join(RUN_DIR, 'result')


def train(env, init_policy, algorithm, params):
    # init policy
    pi = init_policy
    pi.save_model(MODEL_DIR, iteration=0)

    # init params
    with open(params) as params_file:
        d = json.load(params_file)
        num_iters = d.get('num_iters')
        num_ppo_iters = d.get('num_ppo_iters')
        timesteps_per_actorbatch = d.get('timesteps_per_actorbatch')
        clip_param = d.get('clip_param')
        entcoeff = d.get('entcoeff')
        optim_epochs = d.get('optim_epochs')
        optim_stepsize = d.get('optim_stepsize')
        optim_batchsize = d.get('optim_batchsize')
        gamma = d.get('gamma')
        lam = d.get('lam')
        max_iters = num_ppo_iters

    # record performance data
    overall_perf = list()
    ppo_reward = list()
    ppo_length = list()

    # index for num_iters loop
    i = 1
    while i <= num_iters:
        print('overall training iteration %d' %i)
        # each learning step contains "num_ppo_iters" ppo-learning steps.
        # each ppo-learning steps == ppo-learning on single episode
        # each single episode is a single markov chain which contains many states, actions, rewards.
        pi, ep_mean_length, ep_mean_reward = algorithm.ppo_learn(env=env, policy=pi, timesteps_per_actorbatch=timesteps_per_actorbatch,
                                                                 clip_param=clip_param, entcoeff=entcoeff, optim_epochs=optim_epochs,
                                                                 optim_stepsize=optim_stepsize, optim_batchsize=optim_batchsize,
                                                                 gamma=gamma, lam=lam, max_iters=max_iters)

        ppo_length.extend(ep_mean_length)
        ppo_reward.extend(ep_mean_reward)

        # perf_metric = evaluate()
        # overall_perf.append(perf_metric)
        # print('[Overall Iter %d]: perf_metric = %.2f' % (i, perf_metric))

        # Incrementing our algorithm's loop counter
        i += 1

        pi.save_model(MODEL_DIR, iteration=i)
        plot_performance(range(len(ppo_reward)), ppo_reward, ylabel=r'avg reward per ppo-learning step',
                         xlabel='ppo iteration', figfile=os.path.join(FIGURE_DIR, 'ppo_reward'))

        # save data which is accumulated UNTIL iter i
        with open(RESULT_DIR + '/ppo_length_'+'iter_'+str(i)+'.pickle','wb') as f1:
            pickle.dump(ppo_length, f1)
        with open(RESULT_DIR + '/ppo_reward_'+'iter_'+str(i)+'.pickle','wb') as f2:
            pickle.dump(ppo_reward, f2)

    # plot_performance(range(len(overall_perf)), overall_perf, ylabel=r'overall performance per algorithm step',
    #                  xlabel='algorithm iteration',
    #                  figfile=os.path.join(FIGURE_DIR, 'overall_perf'))

    return pi


if __name__ == "__main__":

    # Make necessary directories
    maybe_mkdir(RUN_DIR)
    maybe_mkdir(MODEL_DIR)
    maybe_mkdir(FIGURE_DIR)
    maybe_mkdir(RESULT_DIR)
    outdir = '/tmp/experiments/'
    params_json = './params.json'

    # For live plot
    plotter = liveplot.LivePlot(outdir)

    # Initialize environment and monitor
    env = gym.make(args.gym_env)
    env.reward_type = args.reward_type
    # env = gym.wrappers.Monitor(env, outdir, force=True)

    # Initialize brs engine. You also have to call reset_variables() after instance initialization
    if args.reward_type == 'ttr':

        if args.gym_env == 'DubinsCarEnv-v0':
            brsEngine = DubinsCar_brs_engine()
            brsEngine.reset_variables()

            # test = json.dumps(brsEngine.__dict__)

        elif args.gym_env == 'PlanarQuadEnv-v0':
            brsEngine = Quadrotor_brs_engine()
            brsEngine.reset_variables()

            # test = json.dumps(brsEngine.__dict__)
        else:
            raise ValueError("invalid environment name for ttr reward!")

        # You have to assign the engine
        env.brsEngine = brsEngine

    elif args.reward_type == 'hand_craft':
        pass
    else:
        raise ValueError("wrong type of reward")

    # Initialize policy
    ppo.create_session()
    init_policy = ppo.create_policy('pi', env)
    ppo.initialize()

    # Start to train the policy
    trained_policy = train(env=env, init_policy=init_policy, algorithm=ppo, params=params_json)
    trained_policy.save_model(MODEL_DIR)

    # LOAD_DIR = "./runs/DubinsCarEnv-v0_hand_craft_29-Jan-2019_20-41-17/model"
    # init_policy.load_model(LOAD_DIR, iteration=2)
    # trained_policy = train(env=env, init_policy=init_policy, algorithm=ppo, params=params_json)



