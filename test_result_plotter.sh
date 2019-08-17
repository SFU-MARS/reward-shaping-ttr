#!/usr/bin/env bash

BASEDIR=$(dirname "$0")

#python3.5 $BASEDIR/results_plotter_paper.py -results_path \
#/local-scratch/xlv/reward_shaping_ttr/runs_icra/PlanarQuadEnv-v0_ttr_ppo_02-Aug-2019_12-50-54 \
#/local-scratch/xlv/reward_shaping_ttr/runs_icra/PlanarQuadEnv-v0_hand_craft_ppo_02-Aug-2019_22-51-33 \
#/local-scratch/xlv/reward_shaping_ttr/runs_icra/PlanarQuadEnv-v0_distance_ppo_02-Aug-2019_16-24-50 \
#/local-scratch/xlv/reward_shaping_ttr/runs_icra/PlanarQuadEnv-v0_distance_lambda_10_ppo_03-Aug-2019_11-24-09 \
#/local-scratch/xlv/reward_shaping_ttr/runs_icra/PlanarQuadEnv-v0_distance_lambda_1_ppo_03-Aug-2019_21-23-33 \
#/local-scratch/xlv/reward_shaping_ttr/runs_icra/PlanarQuadEnv-v0_distance_lambda_0.1_ppo_04-Aug-2019_06-13-35 \
#-results_iter 9 20 20 20 18 20 \
#-title Quadrotor_PPO
#
#
#
python3.5 $BASEDIR/results_plotter_paper.py -results_path \
/local-scratch/xlv/reward_shaping_ttr/runs_icra/PlanarQuadEnv-v0_ttr_ddpg_01-Aug-2019_18-13-27 \
/local-scratch/xlv/reward_shaping_ttr/runs_icra/PlanarQuadEnv-v0_hand_craft_ddpg_03-Aug-2019_07-53-57 \
/local-scratch/xlv/reward_shaping_ttr/runs_icra/PlanarQuadEnv-v0_distance_ddpg_03-Aug-2019_04-44-27 \
/local-scratch/xlv/reward_shaping_ttr/runs_icra/PlanarQuadEnv-v0_distance_lambda_10_ddpg_03-Aug-2019_16-45-29 \
/local-scratch/xlv/reward_shaping_ttr/runs_icra/PlanarQuadEnv-v0_distance_lambda_1_ddpg_04-Aug-2019_03-03-57 \
/local-scratch/xlv/reward_shaping_ttr/runs_icra/PlanarQuadEnv-v0_distance_lambda_0.1_ddpg_04-Aug-2019_11-27-04 \
-results_iter 9 9 9 9 9 9 \
-title Quadrotor_DDPG

python3.5 $BASEDIR/results_plotter_paper.py -results_path \
/local-scratch/xlv/reward_shaping_ttr/runs_icra/PlanarQuadEnv-v0_ttr_trpo_06-Aug-2019_23-31-12 \
/local-scratch/xlv/reward_shaping_ttr/runs_icra/PlanarQuadEnv-v0_hand_craft_trpo_07-Aug-2019_03-58-59 \
/local-scratch/xlv/reward_shaping_ttr/runs_icra/PlanarQuadEnv-v0_distance_trpo_07-Aug-2019_09-03-27 \
/local-scratch/xlv/reward_shaping_ttr/runs_icra/PlanarQuadEnv-v0_distance_lambda_1_trpo_07-Aug-2019_19-19-06 \
/local-scratch/xlv/reward_shaping_ttr/runs_icra/PlanarQuadEnv-v0_distance_lambda_0.1_trpo_07-Aug-2019_15-13-54 \
/local-scratch/xlv/reward_shaping_ttr/runs_icra/PlanarQuadEnv-v0_distance_lambda_10_trpo_09-Aug-2019_14-30-51 \
-results_iter 12 20 20 20 20 20 \
-title Quadrotor_TRPO