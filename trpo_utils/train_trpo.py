#!/local-scratch/xlv/miniconda3/envs/py35_no_specific/bin/python

import gym
from time import *
from baselines import logger
from utils.utils import *

from brs_engine.DubinsCar_brs_engine import *
from brs_engine.PlanarQuad_brs_engine import *

from trpo_utils import trpo_mpi, trpo


def train(env, algorithm, params=None, load=False, loadpath=None, loaditer=None):
    assert algorithm == trpo
    assert args.gym_env == "DubinsCarEnv-v0" or args.gym_env == "PlanarQuadEnv-v0"

    trpo.create_session()
    init_policy = trpo.create_policy(env, 'mlp', value_network='copy')
    trpo.initialize()

    if load and loadpath is not None and loaditer is not None:
        # load trained policy
        pi = init_policy
        pi.load_model(loadpath, iteration=loaditer)
        pi.save_model(MODEL_DIR, iteration=0)
    else:
        # init policy
        pi = init_policy
        pi.save_model(MODEL_DIR, iteration=0)

    # record performance data
    overall_perf = list()
    trpo_reward = list()
    trpo_length = list()
    suc_percents = list()
    wall_clock_time = list()

    best_suc_percent = 0
    best_pi = None
    perf_flag = False

    eval_trpo_reward = list()
    eval_suc_percents = list()
    # index for num_iters loop
    i = 1
    num_iters = 10
    max_iters = 30
    while i <= num_iters:
        wall_clock_time.append(time())
        logger.info('overall training iteration %d' % i)

        pi, ep_mean_length, ep_mean_reward, suc_percent = algorithm.trpo_learn(env=env, policy=pi,
                                                                               timesteps_per_batch=1024,
                                                                               # what to train on
                                                                               max_kl=0.001,
                                                                               cg_iters=10,
                                                                               gamma=0.99,
                                                                               lam=1.0,  # advantage estimation
                                                                               seed=None,
                                                                               entcoeff=0.0,
                                                                               cg_damping=1e-2,
                                                                               vf_stepsize=3e-4,
                                                                               vf_iters=3,
                                                                               max_timesteps=0, max_episodes=0,
                                                                               max_iters=max_iters,  # time constraint
                                                                               callback=None,
                                                                               load_path=None)

        trpo_length.extend(ep_mean_length)
        trpo_reward.extend(ep_mean_reward)
        suc_percents.append(suc_percent)


        pi.save_model(MODEL_DIR, iteration=i)
        plot_performance(range(len(trpo_reward)), trpo_reward, ylabel=r'avg reward per trpo-learning step',
                         xlabel='trpo iteration', figfile=os.path.join(FIGURE_DIR, 'trpo_reward'), title='TRAIN')
        plot_performance(range(len(suc_percents)), suc_percents,
                         ylabel=r'overall success percentage per algorithm step of trpo',
                         xlabel='algorithm iteration', figfile=os.path.join(FIGURE_DIR, 'success_percent'),
                         title="TRAIN")

        # --- for plotting evaluation perf on success rate using early stopping trick ---
        if suc_percent > best_suc_percent:
            best_suc_percent = suc_percent
            best_pi = pi

        # TODO: check the logit here
        if suc_percent > 0.6:
            perf_flag = True
        if not perf_flag:
            # less timesteps_per_actorbatch to make eval faster.
            _, _, eval_ep_mean_reward, eval_suc_percent = algorithm.trpo_eval(env, pi, timesteps_per_actorbatch // 2,
                                                                             max_iters=5, stochastic=False)
        else:
            _, _, eval_ep_mean_reward, eval_suc_percent = algorithm.trpo_eval(env, best_pi,
                                                                             timesteps_per_actorbatch // 2,
                                                                             max_iters=5, stochastic=False)
        eval_trpo_reward.extend(eval_ep_mean_reward)
        eval_suc_percents.append(eval_suc_percent)

        plot_performance(range(len(eval_trpo_reward)), eval_trpo_reward, ylabel=r'avg reward per trpo-eval step',
                         xlabel='trpo iteration', figfile=os.path.join(FIGURE_DIR, 'eval_trpo_reward'), title='EVAL')
        plot_performance(range(len(eval_suc_percents)), eval_suc_percents,
                         ylabel=r'overall eval success percentage per algorithm step of trpo',
                         xlabel='algorithm iteration', figfile=os.path.join(FIGURE_DIR, 'eval_success_percent'),
                         title="EVAL")
        # -------------------------------------------------------------------------------

        # save data which is accumulated UNTIL iter i
        with open(RESULT_DIR + '/trpo_length_' + 'iter_' + str(i) + '.pickle', 'wb') as f1:
            pickle.dump(trpo_length, f1)
        with open(RESULT_DIR + '/trpo_reward_' + 'iter_' + str(i) + '.pickle', 'wb') as f2:
            pickle.dump(trpo_reward, f2)
        with open(RESULT_DIR + '/success_percent_' + 'iter_' + str(i) + '.pickle', 'wb') as fs:
            pickle.dump(suc_percents, fs)
        with open(RESULT_DIR + '/wall_clock_time_' + 'iter_' + str(i) + '.pickle', 'wb') as ft:
            pickle.dump(wall_clock_time, ft)

        # save evaluation data accumulated until iter i
        with open(RESULT_DIR + '/eval_trpo_reward_' + 'iter_' + str(i) + '.pickle', 'wb') as f_er:
            pickle.dump(eval_trpo_reward, f_er)
        with open(RESULT_DIR + '/eval_success_percent_' + 'iter_' + str(i) + '.pickle', 'wb') as f_es:
            pickle.dump(eval_suc_percents, f_es)

        # Incrementing our algorithm's loop counter
        i += 1

    # TODO: overall, we need plot the time-to-reach for the best policy so far.

    env.close()

    return pi


if __name__ == "__main__":
    # TRPO optimizing for current task
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--gym_env', type=str, default='PlanarQuadEnv-v0')
    parser.add_argument('--reward_type', type=str, default='hand_craft')
    parser.add_argument('--algo', type=str, default='trpo')
    parser.add_argument('--set_additional_goal', type=str, default='None')
    args = parser.parse_args()

    # ----- path setting ------
    RUN_DIR = MODEL_DIR = FIGURE_DIR = RESULT_DIR = None
    RUN_DIR = os.path.join(os.getcwd(), 'runs_icra',
                           args.gym_env + '_' + args.reward_type + '_' + args.algo + '_' + strftime(
                               '%d-%b-%Y_%H-%M-%S'))
    MODEL_DIR = os.path.join(RUN_DIR, 'model')
    FIGURE_DIR = os.path.join(RUN_DIR, 'figure')
    RESULT_DIR = os.path.join(RUN_DIR, 'result')
    # ---------------------------

    # ------- logger initialize and configuration -------
    logger.configure(dir=RUN_DIR)
    # ---------------------------------------------------

    kwargs = {'reward_type':args.reward_type, 'set_additional_goal':args.set_additional_goal}
    env = gym.make(args.gym_env)
    env.reward_type= args.reward_type
    env.set_additional_goal= args.set_additional_goal

    logger.record_tabular("env", args.gym_env)
    logger.record_tabular("env.set_additional_goal", env.set_additional_goal)
    logger.record_tabular("env.reward_type", env.reward_type)
    logger.record_tabular("algo", args.algo)
    logger.dump_tabular()

    # --- Initialize brs engine. You also have to call reset_variables() after instance initialization ---
    if args.reward_type == 'ttr':

        if args.gym_env == 'DubinsCarEnv-v0':
            brsEngine = DubinsCar_brs_engine()
            brsEngine.reset_variables()

        elif args.gym_env == 'PlanarQuadEnv-v0':
            brsEngine = Quadrotor_brs_engine()
            brsEngine.reset_variables()

        else:
            raise ValueError("invalid environment name for ttr reward!")

        # You have to assign the engine!
        env.brsEngine = brsEngine

    elif args.reward_type == 'hand_craft' or arg.reward_type == 'distance':
        pass
    else:
        raise ValueError("wrong type of reward")
    # ----------------------------------------------------------------------------------------------------

    # make necessary directories
    maybe_mkdir(RUN_DIR)
    maybe_mkdir(MODEL_DIR)
    maybe_mkdir(FIGURE_DIR)
    maybe_mkdir(RESULT_DIR)

    # ------------------ Start to train the policy -----------------------
    # trained_policy = train(env=env, algorithm=trpo)
    # trained_policy.save_model(MODEL_DIR)
    # --------------------------------------------------------------------

    trained_policy = trpo_mpi.learn(network='mlp', env=env, total_timesteps=1e6)
    trained_policy.save_model(MODEL_DIR)

