#!/usr/bin/env bash

BASEDIR=$(dirname "$0")

#python3.5 $BASEDIR/results_plotter_paper.py -results_path \
#/local-scratch/xlv/reward_shaping_ttr/runs_icra/PlanarQuadEnv-v0_ttr_ppo_02-Aug-2019_12-50-54 \
#/local-scratch/xlv/reward_shaping_ttr/runs_icra/PlanarQuadEnv-v0_ttr_ppo_14-Sep-2019_09-46-24 \
#/local-scratch/xlv/reward_shaping_ttr/runs_icra/PlanarQuadEnv-v0_hand_craft_ppo_02-Aug-2019_22-51-33 \
#/local-scratch/xlv/reward_shaping_ttr/runs_icra/PlanarQuadEnv-v0_hand_craft_ppo_15-Aug-2019_07-46-54 \
#/local-scratch/xlv/reward_shaping_ttr/runs_icra/PlanarQuadEnv-v0_distance_ppo_02-Aug-2019_16-24-50 \
#/local-scratch/xlv/reward_shaping_ttr/runs_icra/PlanarQuadEnv-v0_distance_lambda_10_ppo_03-Aug-2019_11-24-09 \
#/local-scratch/xlv/reward_shaping_ttr/runs_icra/PlanarQuadEnv-v0_distance_lambda_1_ppo_03-Aug-2019_21-23-33 \
#/local-scratch/xlv/reward_shaping_ttr/runs_icra/PlanarQuadEnv-v0_distance_lambda_0.1_ppo_04-Aug-2019_06-13-35 \
#/local-scratch/xlv/reward_shaping_ttr/runs_icra/PlanarQuadEnv-v0_distance_lambda_0.1_ppo_20-Aug-2019_22-57-02 \
#/local-scratch/xlv/reward_shaping_ttr/runs_icra/PlanarQuadEnv-v0_distance_lambda_0.1_ppo_14-Sep-2019_15-09-03 \
#/local-scratch/xlv/reward_shaping_ttr/runs_icra/PlanarQuadEnv-v0_distance_lambda_10_ppo_21-Aug-2019_05-43-12 \
#-results_iter 9 20 20 20 20 20 18 20 20 20 20 \
#-title Quadrotor_PPO
##
##
##
#python3.5 $BASEDIR/results_plotter_paper.py -results_path \
#/local-scratch/xlv/reward_shaping_ttr/runs_icra/PlanarQuadEnv-v0_ttr_ddpg_01-Aug-2019_18-13-27 \
#/local-scratch/xlv/reward_shaping_ttr/runs_icra/PlanarQuadEnv-v0_hand_craft_ddpg_03-Aug-2019_07-53-57 \
#/local-scratch/xlv/reward_shaping_ttr/runs_icra/PlanarQuadEnv-v0_distance_ddpg_03-Aug-2019_04-44-27 \
#/local-scratch/xlv/reward_shaping_ttr/runs_icra/PlanarQuadEnv-v0_distance_lambda_10_ddpg_03-Aug-2019_16-45-29 \
#/local-scratch/xlv/reward_shaping_ttr/runs_icra/PlanarQuadEnv-v0_distance_lambda_1_ddpg_04-Aug-2019_03-03-57 \
#/local-scratch/xlv/reward_shaping_ttr/runs_icra/PlanarQuadEnv-v0_distance_lambda_0.1_ddpg_04-Aug-2019_11-27-04 \
#-results_iter 9 9 9 9 9 9 \
#-title Quadrotor_DDPG
#
#python3.5 $BASEDIR/results_plotter_paper.py -results_path \
#/local-scratch/xlv/reward_shaping_ttr/runs_icra/PlanarQuadEnv-v0_ttr_trpo_13-Aug-2019_22-25-06 \
#/local-scratch/xlv/reward_shaping_ttr/runs_icra/PlanarQuadEnv-v0_hand_craft_trpo_07-Aug-2019_03-58-59 \
#/local-scratch/xlv/reward_shaping_ttr/runs_icra/PlanarQuadEnv-v0_distance_trpo_07-Aug-2019_09-03-27 \
#/local-scratch/xlv/reward_shaping_ttr/runs_icra/PlanarQuadEnv-v0_distance_lambda_1_trpo_07-Aug-2019_19-19-06 \
#/local-scratch/xlv/reward_shaping_ttr/runs_icra/PlanarQuadEnv-v0_distance_lambda_0.1_trpo_07-Aug-2019_15-13-54 \
#/local-scratch/xlv/reward_shaping_ttr/runs_icra/PlanarQuadEnv-v0_distance_lambda_10_trpo_09-Aug-2019_14-30-51 \
#-results_iter 20 20 20 20 20 20 \
#-title Quadrotor_TRPO


#python3.5 $BASEDIR/results_plotter_paper.py -results_path \
#/local-scratch/xlv/reward_shaping_ttr/runs_icra/DubinsCarEnv-v0_ttr_ppo_19-Aug-2019_15-16-14 \
#/local-scratch/xlv/reward_shaping_ttr/runs_icra/DubinsCarEnv-v0_ttr_ppo_14-Sep-2019_02-45-25 \
#/local-scratch/xlv/reward_shaping_ttr/runs_icra/DubinsCarEnv-v0_hand_craft_ppo_19-Aug-2019_19-24-18 \
#/local-scratch/xlv/reward_shaping_ttr/runs_icra/DubinsCarEnv-v0_distance_ppo_19-Aug-2019_23-26-10 \
#/local-scratch/xlv/reward_shaping_ttr/runs_icra/DubinsCarEnv-v0_distance_lambda_0.1_ppo_20-Aug-2019_03-26-21 \
#/local-scratch/xlv/reward_shaping_ttr/runs_icra/DubinsCarEnv-v0_distance_lambda_1_ppo_20-Aug-2019_07-32-31 \
#/local-scratch/xlv/reward_shaping_ttr/runs_icra/DubinsCarEnv-v0_distance_lambda_10_ppo_20-Aug-2019_17-12-54 \
#-results_iter 20 10 20 20 20 20 20 \
#-title SimpleCar_PPO
#
#python3.5 $BASEDIR/results_plotter_paper.py -results_path \
#/local-scratch/xlv/reward_shaping_ttr/runs_icra/DubinsCarEnv-v0_ttr_trpo_30-Aug-2019_21-53-13 \
#/local-scratch/xlv/reward_shaping_ttr/runs_icra/DubinsCarEnv-v0_ttr_trpo_13-Sep-2019_19-32-55 \
#/local-scratch/xlv/reward_shaping_ttr/runs_icra/DubinsCarEnv-v0_hand_craft_trpo_31-Aug-2019_11-46-07 \
#/local-scratch/xlv/reward_shaping_ttr/runs_icra/DubinsCarEnv-v0_distance_trpo_31-Aug-2019_18-10-27 \
#/local-scratch/xlv/reward_shaping_ttr/runs_icra/DubinsCarEnv-v0_distance_lambda_0.1_trpo_01-Sep-2019_00-36-23 \
#/local-scratch/xlv/reward_shaping_ttr/runs_icra/DubinsCarEnv-v0_distance_lambda_1_trpo_13-Sep-2019_22-58-50 \
#/local-scratch/xlv/reward_shaping_ttr/runs_icra/DubinsCarEnv-v0_distance_lambda_1_trpo_01-Sep-2019_07-02-13 \
#/local-scratch/xlv/reward_shaping_ttr/runs_icra/DubinsCarEnv-v0_distance_lambda_10_ppo_20-Aug-2019_17-12-54 \
#-results_iter 20 3 20 20 20 10 20 20 \
#-title SimpleCar_TRPO
#
#
#python3.5 $BASEDIR/results_plotter_paper.py -results_path \
#/local-scratch/xlv/reward_shaping_ttr/runs_icra/DubinsCarEnv-v0_ttr_ddpg_18-Aug-2019_04-21-17 \
#/local-scratch/xlv/reward_shaping_ttr/runs_icra/DubinsCarEnv-v0_hand_craft_ddpg_01-Sep-2019_22-54-52 \
#/local-scratch/xlv/reward_shaping_ttr/runs_icra/DubinsCarEnv-v0_distance_ddpg_02-Sep-2019_07-38-39 \
#/local-scratch/xlv/reward_shaping_ttr/runs_icra/DubinsCarEnv-v0_distance_lambda_10_ddpg_03-Sep-2019_09-45-13 \
#/local-scratch/xlv/reward_shaping_ttr/runs_icra/DubinsCarEnv-v0_distance_lambda_1_ddpg_03-Sep-2019_00-57-53 \
#/local-scratch/xlv/reward_shaping_ttr/runs_icra/DubinsCarEnv-v0_distance_lambda_1_ddpg_08-Sep-2019_09-11-57 \
#/local-scratch/xlv/reward_shaping_ttr/runs_icra/DubinsCarEnv-v0_distance_lambda_0.1_ddpg_02-Sep-2019_16-14-25 \
#/local-scratch/xlv/reward_shaping_ttr/runs_icra/DubinsCarEnv-v0_distance_lambda_0.1_ddpg_07-Sep-2019_23-10-22 \
#-results_iter 20 20 20 20 20 20 20 20 20 \
#-title SimpleCar_DDPG











# only show TTR reward curves
python3.5 $BASEDIR/results_plotter_paper.py -results_path \
/local-scratch/xlv/reward_shaping_ttr/runs_icra/PlanarQuadEnv-v0_ttr_ppo_02-Aug-2019_12-50-54 \
/local-scratch/xlv/reward_shaping_ttr/runs_icra/PlanarQuadEnv-v0_ttr_ppo_14-Sep-2019_09-46-24 \
-results_iter 9 20 \
-title Quadrotor_PPO

python3.5 $BASEDIR/results_plotter_paper.py -results_path \
/local-scratch/xlv/reward_shaping_ttr/runs_icra/PlanarQuadEnv-v0_ttr_ddpg_01-Aug-2019_18-13-27 \
-results_iter 9 \
-title Quadrotor_DDPG

python3.5 $BASEDIR/results_plotter_paper.py -results_path \
/local-scratch/xlv/reward_shaping_ttr/runs_icra/PlanarQuadEnv-v0_ttr_trpo_13-Aug-2019_22-25-06 \
-results_iter 20 \
-title Quadrotor_TRPO

# show TTR and sparse curves
python3.5 $BASEDIR/results_plotter_paper.py -results_path \
/local-scratch/xlv/reward_shaping_ttr/runs_icra/PlanarQuadEnv-v0_ttr_ppo_02-Aug-2019_12-50-54 \
/local-scratch/xlv/reward_shaping_ttr/runs_icra/PlanarQuadEnv-v0_ttr_ppo_14-Sep-2019_09-46-24 \
/local-scratch/xlv/reward_shaping_ttr/runs_icra/PlanarQuadEnv-v0_hand_craft_ppo_02-Aug-2019_22-51-33 \
/local-scratch/xlv/reward_shaping_ttr/runs_icra/PlanarQuadEnv-v0_hand_craft_ppo_15-Aug-2019_07-46-54 \
-results_iter 9 20 20 20 \
-title Quadrotor_PPO



python3.5 $BASEDIR/results_plotter_paper.py -results_path \
/local-scratch/xlv/reward_shaping_ttr/runs_icra/PlanarQuadEnv-v0_ttr_ddpg_01-Aug-2019_18-13-27 \
/local-scratch/xlv/reward_shaping_ttr/runs_icra/PlanarQuadEnv-v0_hand_craft_ddpg_03-Aug-2019_07-53-57 \
-results_iter 9 9 \
-title Quadrotor_DDPG


python3.5 $BASEDIR/results_plotter_paper.py -results_path \
/local-scratch/xlv/reward_shaping_ttr/runs_icra/PlanarQuadEnv-v0_ttr_trpo_13-Aug-2019_22-25-06 \
/local-scratch/xlv/reward_shaping_ttr/runs_icra/PlanarQuadEnv-v0_hand_craft_trpo_07-Aug-2019_03-58-59 \
-results_iter 20 20 \
-title Quadrotor_TRPO

# show TTR, sparse and distance curves
python3.5 $BASEDIR/results_plotter_paper.py -results_path \
/local-scratch/xlv/reward_shaping_ttr/runs_icra/PlanarQuadEnv-v0_ttr_ppo_02-Aug-2019_12-50-54 \
/local-scratch/xlv/reward_shaping_ttr/runs_icra/PlanarQuadEnv-v0_ttr_ppo_14-Sep-2019_09-46-24 \
/local-scratch/xlv/reward_shaping_ttr/runs_icra/PlanarQuadEnv-v0_hand_craft_ppo_02-Aug-2019_22-51-33 \
/local-scratch/xlv/reward_shaping_ttr/runs_icra/PlanarQuadEnv-v0_hand_craft_ppo_15-Aug-2019_07-46-54 \
/local-scratch/xlv/reward_shaping_ttr/runs_icra/PlanarQuadEnv-v0_distance_ppo_02-Aug-2019_16-24-50 \
/local-scratch/xlv/reward_shaping_ttr/runs_icra/PlanarQuadEnv-v0_distance_lambda_10_ppo_03-Aug-2019_11-24-09 \
/local-scratch/xlv/reward_shaping_ttr/runs_icra/PlanarQuadEnv-v0_distance_lambda_1_ppo_03-Aug-2019_21-23-33 \
/local-scratch/xlv/reward_shaping_ttr/runs_icra/PlanarQuadEnv-v0_distance_lambda_0.1_ppo_04-Aug-2019_06-13-35 \
/local-scratch/xlv/reward_shaping_ttr/runs_icra/PlanarQuadEnv-v0_distance_lambda_0.1_ppo_20-Aug-2019_22-57-02 \
/local-scratch/xlv/reward_shaping_ttr/runs_icra/PlanarQuadEnv-v0_distance_lambda_0.1_ppo_14-Sep-2019_15-09-03 \
/local-scratch/xlv/reward_shaping_ttr/runs_icra/PlanarQuadEnv-v0_distance_lambda_10_ppo_21-Aug-2019_05-43-12 \
-results_iter 9 20 20 20 20 20 18 20 20 20 20 \
-title Quadrotor_PPO
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
/local-scratch/xlv/reward_shaping_ttr/runs_icra/PlanarQuadEnv-v0_ttr_trpo_13-Aug-2019_22-25-06 \
/local-scratch/xlv/reward_shaping_ttr/runs_icra/PlanarQuadEnv-v0_hand_craft_trpo_07-Aug-2019_03-58-59 \
/local-scratch/xlv/reward_shaping_ttr/runs_icra/PlanarQuadEnv-v0_distance_trpo_07-Aug-2019_09-03-27 \
/local-scratch/xlv/reward_shaping_ttr/runs_icra/PlanarQuadEnv-v0_distance_lambda_1_trpo_07-Aug-2019_19-19-06 \
/local-scratch/xlv/reward_shaping_ttr/runs_icra/PlanarQuadEnv-v0_distance_lambda_0.1_trpo_07-Aug-2019_15-13-54 \
/local-scratch/xlv/reward_shaping_ttr/runs_icra/PlanarQuadEnv-v0_distance_lambda_10_trpo_09-Aug-2019_14-30-51 \
-results_iter 20 20 20 20 20 20 \
-title Quadrotor_TRPO