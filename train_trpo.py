#!/local-scratch/xlv/miniconda3/envs/py35_no_specific/bin/python

import os
import gym

# import spinup

from trpo_utils import trpo_mpi


def train(env, algorithm, params=None, load=False, loadpath=None, loaditer=None):
    assert algorithm == trpo_mpi
    assert args.gym_env == "DubinsCarEnv-v0" or args.gym_env == "PlanarQuadEnv-v0"
    pass



if __name__ == "__main__":
    # TRPO optimizing for current task
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--gym_env', type=str, default='PlanarQuadEnv-v0')
    parser.add_argument('--reward_type', type=str, default='hand_craft')
    parser.add_argument('--algo', type=str, default='trpo')
    parser.add_argument('--set_angle_goal', type=str, default='false')
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

    kwargs = {'reward_type':args.reward_type, 'set_angle_goal':args.set_angle_goal}
    env = gym.make(args.gym_env)
    env.reward_type= args.reward_type
    env.set_angle_goal= args.set_angle_goal

    # make necessary directories
    maybe_mkdir(RUN_DIR)
    maybe_mkdir(MODEL_DIR)
    maybe_mkdir(FIGURE_DIR)
    maybe_mkdir(RESULT_DIR)

    trained_policy = trpo_mpi.learn(network='mlp', env=env, total_timesteps=1e6)
    trained_policy.save_model(MODEL_DIR)