import sys,os
sys.path.append(os.environ['PROJ_HOME'] + "/stable-baselines")
import gym
from time import *
from baselines import logger
from utils.utils import *

from gym_foo import gym_foo
from brs_engine.DubinsCar_brs_engine import *
from brs_engine.PlanarQuad_brs_engine import *

from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines import TRPO

# env = gym.make('CartPole-v1')
# env = DummyVecEnv([lambda: env])
#
# model = TRPO(MlpPolicy, env, verbose=1)
# model.learn(total_timesteps=25000)
# model.save("trpo_cartpole")
#
# del model # remove to demonstrate saving and loading
#
# model = TRPO.load("trpo_cartpole")
#
# obs = env.reset()
# while True:
#     action, _states = model.predict(obs)
#     obs, rewards, dones, info = env.step(action)
#     env.render()

if __name__ == "__main__":
    # TRPO optimizing for current task
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--gym_env', type=str, default='PlanarQuadEnv-v0')
    parser.add_argument('--reward_type', type=str, default='ttr')
    parser.add_argument('--algo', type=str, default='trpo')
    parser.add_argument('--set_additional_goal', type=str, default='None')
    args = parser.parse_args()
    args = vars(args)

    # ----- path setting ------
    RUN_DIR = MODEL_DIR = FIGURE_DIR = RESULT_DIR = None
    RUN_DIR = os.path.join(os.path.dirname(__file__), 'runs_icra',
                           args['gym_env'] + '_' + args['reward_type'] + '_' + args['algo'] + '_' + strftime(
                               '%d-%b-%Y_%H-%M-%S'))
    # print("RUN_DIR:", RUN_DIR)
    MODEL_DIR = os.path.join(RUN_DIR, 'model')
    FIGURE_DIR = os.path.join(RUN_DIR, 'figure')
    RESULT_DIR = os.path.join(RUN_DIR, 'result')
    # ---------------------------

    # ------- logger initialize and configuration -------
    logger.configure(dir=RUN_DIR)
    # ---------------------------------------------------

    kwargs = {'reward_type':args['reward_type'], 'set_additional_goal':args['set_additional_goal']}
    env = gym.make(args['gym_env'])
    env.reward_type= args['reward_type']
    env.set_additional_goal= args['set_additional_goal']

    logger.record_tabular("env", args['gym_env'])
    logger.record_tabular("env.set_additional_goal", env.set_additional_goal)
    logger.record_tabular("env.reward_type", env.reward_type)
    logger.record_tabular("algo", args['algo'])
    logger.dump_tabular()

    # --- Initialize brs engine. You also have to call reset_variables() after instance initialization ---
    if args['reward_type'] == 'ttr':
        if args['gym_env'] == 'DubinsCarEnv-v0':
            brsEngine = DubinsCar_brs_engine()
            brsEngine.reset_variables()
        elif args['gym_env'] == 'PlanarQuadEnv-v0':
            brsEngine = Quadrotor_brs_engine()
            brsEngine.reset_variables()
        else:
            raise ValueError("invalid environment name for ttr reward!")
        # You have to assign the engine!
        env.brsEngine = brsEngine
    elif args['reward_type'] in ['hand_craft','distance','distance_lambda_10','distance_lambda_1','distance_lambda_0.1']:
        pass
    else:
        raise ValueError("wrong type of reward")
    # ----------------------------------------------------------------------------------------------------

    args['RUN_DIR'] = RUN_DIR
    args['MODEL_DIR'] = MODEL_DIR
    args['FIGURE_DIR'] = FIGURE_DIR
    args['RESULT_DIR'] = RESULT_DIR

    # make necessary directories
    maybe_mkdir(RUN_DIR)
    maybe_mkdir(MODEL_DIR)
    maybe_mkdir(FIGURE_DIR)
    maybe_mkdir(RESULT_DIR)

    model = TRPO(MlpPolicy, env, verbose=1, **args)
    # 600 epochs, each epoch 1024 steps; every 30 epochs, do an evaluation.
    model.learn(total_timesteps=1024*601)
    model.save(MODEL_DIR)