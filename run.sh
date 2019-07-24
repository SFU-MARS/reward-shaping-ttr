#!/usr/bin/env bash

# train
#/usr/bin/python3.5 /home/xlv/Desktop/IROS2019/train_ppo.py --gym_env=DubinsCarEnv-v0 --reward_type=distance --algo=ppo
#/usr/bin/python3.5 /home/xlv/Desktop/IROS2019/train_ppo.py --gym_env=DubinsCarEnv-v0 --reward_type=distance --algo=ppo
#/usr/bin/python3.5 /home/xlv/Desktop/IROS2019/train_ppo.py --gym_env=DubinsCarEnv-v0 --reward_type=distance --algo=ppo
#

#/usr/bin/python3.5 /home/xlv/Desktop/IROS2019/train_ppo.py --gym_env=PlanarQuadEnv-v0 --reward_type=distance --algo=ppo --set_hover_end=true
#/usr/bin/python3.5 /home/xlv/Desktop/IROS2019/train_ppo.py --gym_env=PlanarQuadEnv-v0 --reward_type=distance --algo=ppo --set_hover_end=true

#/usr/bin/python3.5 /home/xlv/Desktop/IROS2019/train_ppo.py --gym_env=PlanarQuadEnv-v0 --reward_type=hand_craft --algo=ppo --set_hover_end=true
#/usr/bin/python3.5 /home/xlv/Desktop/IROS2019/train_ppo.py --gym_env=PlanarQuadEnv-v0 --reward_type=hand_craft --algo=ppo --set_hover_end=true
#

BASEDIR=$(dirname "$0")
# echo "$BASEDIR"
# ------- training agent using PPO algorithm -------
# python3.5 $BASEDIR/train_ppo.py --gym_env=PlanarQuadEnv-v0 --reward_type=ttr --algo=ppo --set_angle_goal=false
# python3.5 $BASEDIR/train_ppo.py --gym_env=DubinsCarEnv-v0 --reward_type=ttr --algo=ppo

# run 5 times for each reward type using ppo algorithm for quadrotor task
for VARIABLE in 1 2 3 4 5
do
    python3.5 $BASEDIR/train_ppo.py --gym_env=PlanarQuadEnv-v0 --reward_type=ttr --algo=ppo --set_angle_goal=false
    python3.5 $BASEDIR/train_ppo.py --gym_env=PlanarQuadEnv-v0 --reward_type=hand_craft --algo=ppo --set_angle_goal=false
    python3.5 $BASEDIR/train_ppo.py --gym_env=PlanarQuadEnv-v0 --reward_type=distance --algo=ppo --set_angle_goal=false
done
# --------------------------------------------------

# ------- training agent using TRPO algorithm -------
# python3.5 $BASEDIR/train_trpo.py --gym_env=PlanarQuadEnv-v0 --reward_type=ttr --set_angle_goal=false
# python3.5 $BASEDIR/train_trpo.py --gym_env=DubinsCarEnv-v0 --reward_type=ttr

# run 5 times for each reward type using trpo algorithm for quadrotor task
for VARIABLE in 1 2 3 4 5
do
    python3.5 $BASEDIR/train_trpo.py --gym_env=PlanarQuadEnv-v0 --reward_type=ttr --algo=trpo --set_angle_goal=false
    python3.5 $BASEDIR/train_trpo.py --gym_env=PlanarQuadEnv-v0 --reward_type=hand_craft --algo=trpo --set_angle_goal=false
    python3.5 $BASEDIR/train_trpo.py --gym_env=PlanarQuadEnv-v0 --reward_type=distance --algo=trpo --set_angle_goal=false
done
# ---------------------------------------------------

# ------- training agent using DDPG algorithm -------
python3.5 $BASEDIR/train_ddpg.py --env-id=PlanarQuadEnv-v0 --reward_type=ttr --set_angle_goal=false
# python3.5 $BASEDIR/train_ddpg.py --env-id=PlanarQuadEnv-v0 --reward_type=ttr
# ---------------------------------------------------






#/usr/bin/python3.5 /local-scratch/xlv/IROS2019/train_ppo.py --gym_env=DubinsCarEnv-v0 --reward_type=distance --algo=ppo
#/usr/bin/python3.5 /home/xlv/Desktop/IROS2019/train_ppo.py --gym_env=PlanarQuadEnv-v0 --reward_type=ttr --algo=ppo --set_hover_end=true
#/usr/bin/python3.5 /home/xlv/Desktop/IROS2019/train_ppo.py --gym_env=DubinsCarEnv_dqn-v0 --reward_type=ttr --algo=dqn

# eval
#/usr/bin/python3.5 /home/xlv/Desktop/IROS2019/eval.py --load_path=/home/xlv/Desktop/IROS2019/runs_paper/PlanarQuadEnv-v0_ttr_24-Feb-2019_20-33-35 --load_iter=10 --feedback_type=hand_craft
#/usr/bin/python3.5 /home/xlv/Desktop/IROS2019/eval.py --load_path=/home/xlv/Desktop/IROS2019/runs_paper/PlanarQuadEnv-v0_ttr_24-Feb-2019_22-18-18 --load_iter=6 --feedback_type=hand_craft
#/usr/bin/python3.5 /home/xlv/Desktop/IROS2019/eval.py --load_path=/home/xlv/Desktop/IROS2019/runs_paper/PlanarQuadEnv-v0_distance_25-Feb-2019_00-02-04 --load_iter=10 --feedback_type=hand_craft
#/usr/bin/python3.5 /home/xlv/Desktop/IROS2019/eval.py --load_path=/home/xlv/Desktop/IROS2019/runs_paper/PlanarQuadEnv-v0_distance_25-Feb-2019_01-41-56 --load_iter=10 --feedback_type=hand_craft
#/usr/bin/python3.5 /home/xlv/Desktop/IROS2019/eval.py --load_path=/home/xlv/Desktop/IROS2019/runs_paper/PlanarQuadEnv-v0_hand_craft_25-Feb-2019_03-22-52 --load_iter=10 --feedback_type=hand_craft
#/usr/bin/python3.5 /home/xlv/Desktop/IROS2019/eval.py --load_path=/home/xlv/Desktop/IROS2019/runs_paper/PlanarQuadEnv-v0_hand_craft_25-Feb-2019_05-03-22 --load_iter=10 --feedback_type=hand_craft



#/usr/bin/python3.5 /home/xlv/Desktop/IROS2019/eval.py --load_path=/home/xlv/Desktop/IROS2019/runs_paper/DubinsCarEnv-v0_ttr_23-Feb-2019_10-33-57 --load_iter=8 --feedback_type=hand_craft
#/usr/bin/python3.5 /home/xlv/Desktop/IROS2019/eval.py --load_path=/home/xlv/Desktop/IROS2019/runs_paper/DubinsCarEnv-v0_ttr_23-Feb-2019_13-31-37 --load_iter=10 --feedback_type=hand_craft
#/usr/bin/python3.5 /home/xlv/Desktop/IROS2019/eval.py --load_path=/home/xlv/Desktop/IROS2019/runs_paper/DubinsCarEnv-v0_ttr_23-Feb-2019_16-25-18 --load_iter=10 --feedback_type=hand_craft
#/usr/bin/python3.5 /home/xlv/Desktop/IROS2019/eval.py --load_path=/home/xlv/Desktop/IROS2019/runs_paper/DubinsCarEnv-v0_hand_craft_23-Feb-2019_23-01-51 --load_iter=10 --feedback_type=hand_craft
#/usr/bin/python3.5 /home/xlv/Desktop/IROS2019/eval.py --load_path=/home/xlv/Desktop/IROS2019/runs_paper/DubinsCarEnv-v0_hand_craft_23-Feb-2019_23-57-09 --load_iter=10 --feedback_type=hand_craft
#/usr/bin/python3.5 /home/xlv/Desktop/IROS2019/eval.py --load_path=/home/xlv/Desktop/IROS2019/runs_paper/DubinsCarEnv-v0_distance_23-Feb-2019_18-27-53 --load_iter=10 --feedback_type=hand_craft
#/usr/bin/python3.5 /home/xlv/Desktop/IROS2019/eval.py --load_path=/home/xlv/Desktop/IROS2019/runs_paper/DubinsCarEnv-v0_distance_23-Feb-2019_19-24-36 --load_iter=10 --feedback_type=hand_craft
#/usr/bin/python3.5 /home/xlv/Desktop/IROS2019/eval.py --load_path=/home/xlv/Desktop/IROS2019/runs_paper/DubinsCarEnv-v0_distance_23-Feb-2019_20-20-26 --load_iter=10 --feedback_type=hand_craft
#
#

#/usr/bin/python3.5 /home/xlv/Desktop/IROS2019/eval.py --load_path=/home/xlv/Desktop/IROS2019/runs_reborn/PlanarQuadEnv-v0_ttr_28-Feb-2019_12-44-01 --load_iter=9 --feedback_type=hand_craft
#/usr/bin/python3.5 /home/xlv/Desktop/IROS2019/eval.py --load_path=/home/xlv/Desktop/IROS2019/runs_reborn/PlanarQuadEnv-v0_ttr_28-Feb-2019_12-44-01 --load_iter=8 --feedback_type=hand_craft
#
#/usr/bin/python3.5 /home/xlv/Desktop/IROS2019/eval.py --load_path=/home/xlv/Desktop/IROS2019/runs_reborn/PlanarQuadEnv-v0_distance_28-Feb-2019_18-43-57 --load_iter=10 --feedback_type=hand_craft
#/usr/bin/python3.5 /home/xlv/Desktop/IROS2019/eval.py --load_path=/home/xlv/Desktop/IROS2019/runs_reborn/PlanarQuadEnv-v0_distance_28-Feb-2019_20-17-31 --load_iter=10 --feedback_type=hand_craft
