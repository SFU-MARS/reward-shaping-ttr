# reward-shaping-ttr
This repo focus on the research topic about reward shaping via optimal control theory


## Basic Requirement
1.  Install miniconda3 (or anaconda if you like). 
2.  Create a new conda environment and do all the following steps in this environment. 
```
conda create --name [env_name] python=3.5
conda activate [env_name]
```
3.  Install missing python-packages, such as gym, numpy, tensorflow(or tensorflow-gpu), matplotlib, joblib, keras, mpi4py, cycler,
```
conda install [pkg_name]
pip install -U [pkg_name]
```
4.  Install Gazebo8, ROS kinetic, MATLAB2019a for you computer.
5.  Install gazebo-ros from gazebo website. Note the matching between gazebo version and ROS version. Here I use gazebo8 and ROS kinetic.

## Trouble Shooting
1.  No module named rospy
```
    cd $CONDA_PREFIX
	mkdir -p ./etc/conda/activate.d
	mkdir -p ./etc/conda/deactivate.d
	touch ./etc/conda/activate.d/env_vars.sh
	touch ./etc/conda/deactivate.d/env_vars.sh
```
And edit env_vars.sh:
```
#!/bin/sh
export PYTHONPATH=$PYTHONPATH:/opt/ros/kinetic/lib/python2.7/dist-packages
```
2.  No module named rospkg
```
pip install -U rosdep rosinstall_generator wstool rosinstall six vcstools
```
3.  ImportError: dynamic module does not define module export function (PyInit__tf2)

   + First, create catkin_ws folder at whereever you want: 
   ```
   mkdir catkin_ws
   cd catkin_ws
   mkdir src
   cd src
   ```
   + Second, download ROS package tf2, tf, hector_sensor_description from github into src folder

   + Third, re-build these packages using catkin build command:
   ```
   catkin build -DPYTHON_EXECUTABLE=path_to_miniconda/envs/[env_name]/bin/python3.5
   ```
   + Finally,
   Edit $CONDA_PREFIX/etc/conda/activate.d/env_vars.sh again:
   ```
   source /opt/ros/kinetic/setup.bash
   source path_to_catkin_ws/catkin_ws/devel/setup.bash
   ```
4.  sys/os/glnxa64/libstdc++.so.6: version `CXXABI_1.3.11' not found
```
cd  path_to_matlab/MATLAB/R2018b/sys/os/glnxa64
mkdir exclude
mv libstdc++.so.6* exclude/
ln -s /usr/lib/x86_64-linux-gnu/libstdc++.so.6 libstdc++.so.6
ln -s /usr/lib/x86_64-linux-gnu/libstdc++.so.6.0.xx libstdc++.so.6.0.xx
```
5.  If you meet problems with xacro, saying no module named "glob" or "re", you need download xacro from github into catkin_ws src folder and compile it using python3.5 manully.
6.  Everytime you build something new in catkin_ws, remember to source the env_vars again.
7.  For this project, you also need edit on env_vars.sh like:
```
export PROJ_HOME=/local-scratch/xlv/reward_shaping_ttr
export QUADROTOR_WORLD_AIR_SPACE=$PROJ_HOME/worlds/air_space.world
export DUBINS_CAR_WORLD_CIRCUIT_GROUND=$PROJ_HOME/worlds/circuit_ground.world

export ROS_HOSTNAME=localhost
export ROS_MASTER_URI=http://localhost:11311

alias killgazebogym='killall -9 rosout roslaunch rosmaster gzserver nodelet robot_state_publisher gzclient'
```


## Additional tips
1.  After you clone this repo to your local, here is a few things you may need do
   + change the $PROJ_HOME in env_vars.sh
   ```
   $PROJ_HOME = 'path you place this repo at'
   ```
   + you may need to recompile catkin_ws folder to let it recognize your filepath
   + everytime you build something new in catkin_ws or modify something in env_vars, remember
   ```
   source env_vars.sh
   ```
2. stay tuned...
































