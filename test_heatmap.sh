#!/usr/bin/env bash

BASEDIR=$(dirname "$0")

#python3.5 $BASEDIR/heatmap.py --gym_env=PlanarQuadEnv-v0 --reward_type=ttr --algo=ppo --set_additional_goal=angle
#python3.5 $BASEDIR/heatmap.py --gym_env=PlanarQuadEnv-v0 --reward_type=hand_craft --algo=ppo --set_additional_goal=angle
#python3.5 $BASEDIR/heatmap.py --gym_env=PlanarQuadEnv-v0 --reward_type=distance --algo=ppo --set_additional_goal=angle
#python3.5 $BASEDIR/heatmap.py --gym_env=PlanarQuadEnv-v0 --reward_type=distance_lambda_0.1 --algo=ppo --set_additional_goal=angle
#python3.5 $BASEDIR/heatmap.py --gym_env=PlanarQuadEnv-v0 --reward_type=distance_lambda_1 --algo=ppo --set_additional_goal=angle
#python3.5 $BASEDIR/heatmap.py --gym_env=PlanarQuadEnv-v0 --reward_type=distance_lambda_10 --algo=ppo --set_additional_goal=angle


#python3.5 $BASEDIR/heatmap.py --gym_env=PlanarQuadEnv-v0 --loadpath=/local-scratch/xlv/reward_shaping_ttr/heatmaps_icra/PlanarQuadEnv-v0_ttr_ppo_22-Aug-2019_23-24-20

python3.5 $BASEDIR/heatmap.py --gym_env=PlanarQuadEnv-v0 --loadpath=/local-scratch/xlv/reward_shaping_ttr/heatmaps_icra/PlanarQuadEnv-v0_hand_craft_ppo_23-Aug-2019_00-27-32

#python3.5 $BASEDIR/heatmap.py --gym_env=PlanarQuadEnv-v0 --loadpath=/local-scratch/xlv/reward_shaping_ttr/heatmaps_icra/PlanarQuadEnv-v0_distance_lambda_1_ppo_23-Aug-2019_04-55-48