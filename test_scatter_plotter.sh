#!/usr/bin/env bash

BASEDIR=$(dirname "$0")

#/local-scratch/xlv/miniconda3/envs/py35_no_specific/bin/python3.5  $BASEDIR/scatter_plotter_paper.py \
#--gym_env=PlanarQuadEnv-v0 \
#--reward_type=hand_craft \
#--load_dir=/local-scratch/xlv/reward_shaping_ttr/runs_icra/PlanarQuadEnv-v0_ttr_ppo_02-Aug-2019_12-50-54/model \
#--load_iter=4

#/local-scratch/xlv/miniconda3/envs/py35_no_specific/bin/python3.5  $BASEDIR/scatter_plotter_paper.py \
#--gym_env=PlanarQuadEnv-v0 \
#--reward_type=hand_craft \
#--load_dir=/local-scratch/xlv/reward_shaping_ttr/runs_icra/PlanarQuadEnv-v0_hand_craft_ppo_02-Aug-2019_22-51-33/model \
#--load_iter=4

#/local-scratch/xlv/miniconda3/envs/py35_no_specific/bin/python3.5  $BASEDIR/scatter_plotter_paper.py \
#--gym_env=PlanarQuadEnv-v0 \
#--reward_type=hand_craft \
#--load_dir=/local-scratch/xlv/reward_shaping_ttr/runs_icra/PlanarQuadEnv-v0_distance_ppo_02-Aug-2019_16-24-50/model \
#--load_iter=4
#
#/local-scratch/xlv/miniconda3/envs/py35_no_specific/bin/python3.5  $BASEDIR/scatter_plotter_paper.py \
#--gym_env=PlanarQuadEnv-v0 \
#--reward_type=hand_craft \
#--load_dir=/local-scratch/xlv/reward_shaping_ttr/runs_icra/PlanarQuadEnv-v0_distance_lambda_0.1_ppo_04-Aug-2019_06-13-35/model \
#--load_iter=4
#
#/local-scratch/xlv/miniconda3/envs/py35_no_specific/bin/python3.5  $BASEDIR/scatter_plotter_paper.py \
#--gym_env=PlanarQuadEnv-v0 \
#--reward_type=hand_craft \
#--load_dir=/local-scratch/xlv/reward_shaping_ttr/runs_icra/PlanarQuadEnv-v0_distance_lambda_1_ppo_03-Aug-2019_21-23-33/model \
#--load_iter=4

#/local-scratch/xlv/miniconda3/envs/py35_no_specific/bin/python3.5  $BASEDIR/scatter_plotter_paper.py \
#--gym_env=PlanarQuadEnv-v0 \
#--reward_type=hand_craft \
#--load_dir=/local-scratch/xlv/reward_shaping_ttr/runs_icra/PlanarQuadEnv-v0_distance_lambda_10_ppo_03-Aug-2019_11-24-09/model \
#--load_iter=4

/local-scratch/xlv/miniconda3/envs/py35_no_specific/bin/python3.5  $BASEDIR/scatter_plotter_paper.py \
--gym_env=DubinsCarEnv-v0 \
--reward_type=hand_craft \
--load_dir=/local-scratch/xlv/reward_shaping_ttr/runs_icra/DubinsCarEnv-v0_ttr_ppo_18-Aug-2019_00-25-14/model \
--load_iter=8