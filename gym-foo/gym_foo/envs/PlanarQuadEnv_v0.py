import gym
from gym import spaces
from gym.utils import seeding
import numpy as np
import gazebo_env
from utils import *
from gazebo_msgs.msg import ModelState
from geometry_msgs.msg import Twist, Pose
from geometry_msgs.msg import Wrench

from sensor_msgs.msg import LaserScan

from std_srvs.srv import Empty
from gazebo_msgs.srv import GetModelState
from gazebo_msgs.srv import SetModelState
from gazebo_msgs.srv import ApplyJointEffort
from gazebo_msgs.srv import JointRequest # the type of clear joint effort
from gazebo_msgs.srv import ApplyBodyWrench



import rospy
import time
import copy

# from hector_uav_msgs.srv import EnableMotors

from tf.transformations import euler_from_quaternion, quaternion_from_euler

# # need to be compatitable with model.sdf and world.sdf for custom setting
# # notice: it's not the gazebo pose state, not --> x,y,z,pitch,roll,yaw !!
# GOAL_STATE = np.array([4.0, 0., 9., 0., -0.5, 0.])
# START_STATE = np.array([-2.18232, 0., 3., 0., 0., 0.])
#
# # obstacles position groundtruth 1,2,3,4
# OBSTACLES_POS = [(0, 8.5), (-0.5, 6.), (1.0, 5.5), (1., 1.)]
# # wall 0,1,2,3
# WALLS_POS = [(-5., 5.), (5., 5.), (0.0, 9.85), (0.0, 5.)]
#
#
# class PlanarQuadEnv_v0(gazebo_env.GazeboEnv):
#     def __init__(self):
#         # Launch the simulation with the given launchfile name
#         gazebo_env.GazeboEnv.__init__(self, "QuadrotorAirSpace_v0.launch")
#         # self.vel_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=5)
#
#         self.unpause = rospy.ServiceProxy('/gazebo/unpause_physics', Empty)
#         self.pause = rospy.ServiceProxy('/gazebo/pause_physics', Empty)
#         self.reset_proxy = rospy.ServiceProxy('/gazebo/reset_simulation', Empty)
#         self.get_model_state = rospy.ServiceProxy('/gazebo/get_model_state', GetModelState)
#         self.set_model_state = rospy.ServiceProxy('/gazebo/set_model_state', SetModelState)
#         self.apply_joint_effort = rospy.ServiceProxy('/gazebo/apply_joint_effort', ApplyJointEffort)
#         self.clear_joint_effort = rospy.ServiceProxy('/gazebo/clear_joint_forces', JointRequest)
#         self.force = rospy.ServiceProxy('/gazebo/apply_body_wrench', ApplyBodyWrench)
#
#         # self.enable_moters = rospy.ServiceProxy('/enable_motors', EnableMotors)
#         #
#         # # First you need enable motor and do anything else
#         # rospy.wait_for_service('/enable_motors')
#         # try:
#         #     enable_motors = rospy.ServiceProxy('/enable_motors', EnableMotors)
#         #     res = enable_motors(True)
#         #     if res:
#         #         print("Motors enabled!")
#         #     else:
#         #         print("Failed to enable motors...")
#         # except rospy.ServiceException as e:
#         #     print("Enable service", '/enable_motors', "call failed: %s" % e)
#
#         self._seed()
#
#         self.m = 1.25
#         self.g = 9.81
#         self.num_lasers = 8
#
#         self.Thrustmax = 0.75 * self.m * self.g
#         self.Thrustmin = 0
#
#         self.control_reward_coff = 0.01
#         self.collision_reward = -2 * 200 * self.control_reward_coff * (self.Thrustmax ** 2)
#         self.goal_reward = 1000
#
#         self.start_state = START_STATE
#         self.goal_state = GOAL_STATE
#
#         # state space and action space (MlpPolicy needs these params for input)
#         high_state = np.array([5., 2., 10., 2., np.pi, np.pi / 3])
#         low_state = np.array([-5., -2., 0., -2., -np.pi, -np.pi / 3])
#
#         high_obsrv = np.array([5., 2., 10., 2., np.pi, np.pi / 3] + [5*2] * self.num_lasers)
#         low_obsrv = np.array([-5., -2., 0., -2., -np.pi, -np.pi/3] + [0] * self.num_lasers)
#
#         # controls are two thrusts
#         high_action = np.array([10., 10.])
#         low_action = np.array([8., 8.])
#
#         self.state_space = spaces.Box(low=low_state, high=high_state)
#         self.observation_space = spaces.Box(low=low_obsrv, high=high_obsrv)
#         self.action_space = spaces.Box(low=low_action, high=high_action)
#
#         self.state_dim = 6
#         self.action_dim = 2
#
#         self.goal_pos_tolerance = 1.5
#         self.goal_pos_tolerance = 1.5
#         self.goal_vel_limit = 0.25
#         self.goal_phi_limit = np.pi / 6.
#
#         self.pre_obsrv = None
#         self.reward_type = None
#         self.set_hover_end = None
#         self.brsEngine = None
#
#     def _discretize_laser(self, laser_data, new_ranges):
#
#         discretized_ranges = []
#         # mod = int(len(laser_data.ranges)/new_ranges)
#         # for i, item in enumerate(laser_data.ranges):
#         #     if (i+1) % mod == 0:
#         #         if laser_data.ranges[i] == float('Inf') or np.isinf(laser_data.ranges[i]):
#         #             discretized_ranges.append(10)
#         #         elif np.isnan(laser_data.ranges[i]):
#         #             discretized_ranges.append(0)
#         #         else:
#         #             discretized_ranges.append(int(laser_data.ranges[i]))
#
#         full_ranges = float(len(laser_data.ranges))
#         # print("laser ranges num: %d" % full_ranges)
#
#         for i in range(new_ranges):
#             new_i = int(i * full_ranges // new_ranges + full_ranges // (2 * new_ranges))
#             if laser_data.ranges[new_i] == float('Inf') or np.isinf(laser_data.ranges[new_i]):
#                 discretized_ranges.append(10)
#             elif np.isnan(laser_data.ranges[new_i]):
#                 discretized_ranges.append(0)
#             else:
#                 discretized_ranges.append(int(laser_data.ranges[new_i]))
#
#         return discretized_ranges
#
#     def _in_obst(self, laser_data, dynamic_data):
#         laser_min_range = 0.6
#         collision_min_range = 0.8
#         tmp_x = dynamic_data.pose.position.x
#         tmp_y = dynamic_data.pose.position.y
#         tmp_z = dynamic_data.pose.position.z
#         for idx, item in enumerate(laser_data.ranges):
#             # if laser_min_range > laser_data.ranges[idx] > 0:
#             #     return True
#             if tmp_z <= laser_min_range:
#                 return True
#             # collision detection via groundtruth positions of obstacles
#             elif Euclid_dis((tmp_x, tmp_z), OBSTACLES_POS[0]) < collision_min_range or Euclid_dis((tmp_x, tmp_z), OBSTACLES_POS[1]) < collision_min_range \
#                 or Euclid_dis((tmp_x, tmp_z), OBSTACLES_POS[2]) < collision_min_range or Euclid_dis((tmp_x, tmp_z), OBSTACLES_POS[3]) < collision_min_range\
#                 or np.abs(tmp_x - WALLS_POS[0][0]) < laser_min_range or np.abs(tmp_x - WALLS_POS[1][0]) < laser_min_range \
#                 or np.abs(tmp_z - WALLS_POS[2][1]) < laser_min_range:
#                 return True
#         return False
#
#     def _in_goal(self, state):
#         assert len(state) == self.state_dim
#
#         x = state[0]
#         z = state[2]
#         # print("z pos:", z)
#         vx = state[1]
#         vy = state[3]
#         phi = state[4]
#
#         # just consider pose restriction
#         if self.set_hover_end == 'false':
#             if np.sqrt((x - self.goal_state[0]) ** 2 + (z - self.goal_state[2]) ** 2) <= self.goal_pos_tolerance:
#                 return True
#             else:
#                 return False
#         elif self.set_hover_end == 'true':
#             if np.sqrt((x - self.goal_state[0]) ** 2 + (z - self.goal_state[2]) ** 2) <= self.goal_pos_tolerance \
#                     and (phi - self.goal_state[4] <= 0.):
#                 return True
#             else:
#                 return False
#
#     def get_obsrv(self, laser_data, dynamic_data):
#
#         discretized_laser_data = self._discretize_laser(laser_data, 8)
#
#         # planar quadrotor x position
#         x = dynamic_data.pose.position.x
#         # planar quadrotor z position
#         z = dynamic_data.pose.position.z
#
#         # planar quadrotor velocity at x axis,
#         vx = dynamic_data.twist.linear.x
#         # planar quadrotor velocity at y axis == real world velocity z axis
#         vz = dynamic_data.twist.linear.z
#
#         ox = dynamic_data.pose.orientation.x
#         oy = dynamic_data.pose.orientation.y
#         oz = dynamic_data.pose.orientation.z
#         ow = dynamic_data.pose.orientation.w
#
#         # planar quadrotor pitch angle (along x-axis)
#         _, pitch, _ = euler_from_quaternion([ox, oy, oz, ow])
#
#         # planar quadrotor pitch angular velocity
#         w = dynamic_data.twist.angular.y
#
#         obsrv = [x, vx, z, vz, pitch, w] + discretized_laser_data
#
#         return obsrv
#
#     def reset(self):
#         # Resets the state of the environment and returns an initial observation.
#         rospy.wait_for_service('/gazebo/reset_simulation')
#         try:
#             self.reset_proxy()
#             pose = Pose()
#             pose.position.x = np.random.uniform(low=START_STATE[0] - 0.5, high=START_STATE[0] + 0.5)
#             pose.position.y = self.get_model_state(model_name="quadrotor").pose.position.y
#             pose.position.z = np.random.uniform(low=START_STATE[2] - 0.5, high=START_STATE[2] + 0.5)
#             pitch = np.random.uniform(low=START_STATE[4] - 0.1, high=START_STATE[4] + 0.1)
#             ox, oy, oz, ow = quaternion_from_euler(0.0, pitch, 0.0)
#             pose.orientation.x = ox
#             pose.orientation.y = oy
#             pose.orientation.z = oz
#             pose.orientation.w = ow
#
#             reset_state = ModelState()
#             reset_state.model_name = "quadrotor"
#             reset_state.pose = pose
#             self.set_model_state(reset_state)
#         except rospy.ServiceException as e:
#             print("# Resets the state of the environment and returns an initial observation.")
#
#         # rospy.sleep(5.)
#
#         # Unpause simulation to make observation
#         rospy.wait_for_service('/gazebo/unpause_physics')
#         try:
#             self.unpause()
#         except rospy.ServiceException as e:
#             print("/gazebo/unpause_physics service call failed")
#
#         # read laser data
#         laser_data = None
#         dynamic_data = None
#         while laser_data is None and dynamic_data is None:
#             try:
#                 laser_data = rospy.wait_for_message('/scan', LaserScan, timeout=5)
#                 # dynamic_data = rospy.wait_for_message('/gazebo/model_states', ModelStates, timeout=5)
#                 rospy.wait_for_service("/gazebo/get_model_state")
#                 try:
#                     dynamic_data = self.get_model_state(model_name="quadrotor")
#                 except rospy.ServiceException as e:
#                     print("/gazebo/unpause_physics service call failed")
#             except:
#                 pass
#
#         rospy.wait_for_service('/gazebo/pause_physics')
#         try:
#             self.pause()
#         except rospy.ServiceException as e:
#             print("/gazebo/pause_physics service call failed")
#
#         obsrv = self.get_obsrv(laser_data, dynamic_data)
#         self.pre_obsrv = obsrv
#
#         return np.asarray(obsrv)
#
#     def step(self, action):
#         rospy.wait_for_service('/gazebo/unpause_physics')
#         try:
#             self.unpause()
#         except rospy.ServiceException as e:
#             print("/gazebo/unpause_physics service call failed")
#
#         # clip action
#         # print("original action output from NN:", action)
#         # action = np.clip(action + 9.0, self.action_space.low, self.action_space.high)
#         # print("action after being clipped", action)
#
#         action = action + 8.5
#         if sum(np.isnan(action)) > 0:
#             raise ValueError("Passed in nan to step! Action: " + str(action))
#
#         pre_phi = self.pre_obsrv[4]
#
#         wrench = Wrench()
#         wrench.force.x = (action[0] + action[1]) * np.sin(pre_phi)
#         wrench.force.y = 0
#         # wrench.force.z = action[0] + action[1]
#         wrench.force.z = (action[0] + action[1]) * np.cos(pre_phi)
#         wrench.torque.x = 0
#         wrench.torque.y = (action[0] - action[1]) * 0.4
#         # wrench.torque.y = 1.0
#         wrench.torque.z = 0
#
#         rospy.wait_for_service('/gazebo/apply_body_wrench')
#         self.force(body_name="base_link", reference_frame="world", wrench=wrench, start_time=rospy.Time().now(), duration=rospy.Duration(1))
#
#         # rospy.wait_for_service("/gazebo/apply_joint_effort")
#         # try:
#         #     status_left = self.apply_joint_effort(joint_name='left_joint', effort=action[0],
#         #                                           start_time=rospy.Time().now(), duration=rospy.Duration(10))
#         #     status_right = self.apply_joint_effort(joint_name='right_joint', effort=action[1],
#         #                                            start_time=rospy.Time().now(), duration=rospy.Duration(10))
#         #     rospy.loginfo("status left: %d, %s, %d" % (status_left.success, status_left.status_message, rospy.Time().now().to_sec()))
#         #     print("status right:", status_right.success, status_right.status_message)
#         # except rospy.ServiceException as e:
#         #     print("/gazebo/apply_joint_effort service call failed")
#
#         laser_data = None
#         dynamic_data = None
#         while laser_data is None and dynamic_data is None:
#             try:
#                 laser_data = rospy.wait_for_message('/scan', LaserScan, timeout=5)
#                 # dynamic_data = rospy.wait_for_message('/gazebo/model_states', ModelStates)
#                 rospy.wait_for_service("/gazebo/get_model_state")
#                 try:
#                     dynamic_data = self.get_model_state(model_name="quadrotor")
#                 except rospy.ServiceException as e:
#                     print("/gazebo/unpause_physics service call failed")
#             except:
#                 pass
#
#         # self.clear_joint_effort(joint_name='left_joint')
#         # self.clear_joint_effort(joint_name='right_joint')
#
#         rospy.wait_for_service('/gazebo/pause_physics')
#         try:
#             # resp_pause = pause.call()
#             self.pause()
#         except rospy.ServiceException as e:
#             print("/gazebo/pause_physics service call failed")
#
#         obsrv = self.get_obsrv(laser_data, dynamic_data)
#         self.pre_obsrv = obsrv
#
#         assert self.reward_type is not None
#         reward = 0
#
#         if self.reward_type == 'hand_craft':
#             # reward = -self.control_reward_coff * (action[0] ** 2 + action[1] ** 2)
#             # reward = -1
#             reward = 0
#         elif self.reward_type == 'ttr' and self.brsEngine is not None:
#             # Notice z-axis ttr space is defined from (-5,5), in gazebo it's in (0,10), so you need -5 when you want correct ttr reward
#             ttr_obsrv = copy.deepcopy(obsrv)
#             ttr_obsrv[2] = ttr_obsrv[2] - 5
#             ttr = self.brsEngine.evaluate_ttr(np.reshape(ttr_obsrv[:6], (1, -1)))
#             reward = -ttr
#         elif self.reward_type == 'distance':
#             reward = -(Euclid_dis((obsrv[0], obsrv[2]), (GOAL_STATE[0], GOAL_STATE[2])))
#
#         done = False
#         suc = False
#
#         # 1. when collision happens, done = True
#         if self._in_obst(laser_data, dynamic_data):
#             reward += self.collision_reward
#             done = True
#
#         # 2. In the neighbor of goal state, done is True as well. Only considering velocity and pos
#         if self._in_goal(np.array(obsrv[:6])):
#             reward += self.goal_reward
#             done = True
#             suc = True
#
#         if obsrv[4] > 0.8 or obsrv[4] < -0.8:
#             reward += self.collision_reward * 2
#             done = True
#
#         # 3. Maybe episode length limit is another factor for resetting the robot, stay tuned.
#         # waiting to be implemented
#         # if action[0] >= 10 or action[1] >= 10:
#         #     reward += self.collision_reward
#         #     done = True
#         # ---
#         # rospy.loginfo("reward: %f", reward)
#         # assert reward < 0
#         # print("reward,", reward)
#         return np.asarray(obsrv), reward, done, suc, {}
#
#     def _seed(self, seed=None):
#         self.np_random, seed = seeding.np_random(seed)
#         return [seed]

# FOR TEST NEW WORLD.
# need to be compatitable with model.sdf and world.sdf for custom setting
# notice: it's not the gazebo pose state, not --> x,y,z,pitch,roll,yaw !!
GOAL_STATE = np.array([4.0, 0., 9., 0., 0.75, 0.])
START_STATE = np.array([3.18232, 0., 3., 0., 0., 0.])

# obstacles position groundtruth
OBSTACLES_POS = [(-1.5, 3), (3, 6), (-2, 7)]

# wall 0,1,2,3
WALLS_POS = [(-5., 5.), (5., 5.), (0.0, 9.85), (0.0, 5.)]


class PlanarQuadEnv_v0(gazebo_env.GazeboEnv):
    def __init__(self):
        # Launch the simulation with the given launchfile name
        gazebo_env.GazeboEnv.__init__(self, "QuadrotorAirSpace_v0.launch")
        # self.vel_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=5)

        self.unpause = rospy.ServiceProxy('/gazebo/unpause_physics', Empty)
        self.pause = rospy.ServiceProxy('/gazebo/pause_physics', Empty)
        self.reset_proxy = rospy.ServiceProxy('/gazebo/reset_simulation', Empty)
        self.get_model_state = rospy.ServiceProxy('/gazebo/get_model_state', GetModelState)
        self.set_model_state = rospy.ServiceProxy('/gazebo/set_model_state', SetModelState)
        self.apply_joint_effort = rospy.ServiceProxy('/gazebo/apply_joint_effort', ApplyJointEffort)
        self.clear_joint_effort = rospy.ServiceProxy('/gazebo/clear_joint_forces', JointRequest)
        self.force = rospy.ServiceProxy('/gazebo/apply_body_wrench', ApplyBodyWrench)

        # self.enable_moters = rospy.ServiceProxy('/enable_motors', EnableMotors)
        #
        # # First you need enable motor and do anything else
        # rospy.wait_for_service('/enable_motors')
        # try:
        #     enable_motors = rospy.ServiceProxy('/enable_motors', EnableMotors)
        #     res = enable_motors(True)
        #     if res:
        #         print("Motors enabled!")
        #     else:
        #         print("Failed to enable motors...")
        # except rospy.ServiceException as e:
        #     print("Enable service", '/enable_motors', "call failed: %s" % e)

        self._seed()

        self.m = 1.25
        self.g = 9.81
        self.num_lasers = 8

        self.Thrustmax = 0.75 * self.m * self.g
        self.Thrustmin = 0

        self.control_reward_coff = 0.01
        self.collision_reward = -2 * 200 * self.control_reward_coff * (self.Thrustmax ** 2)
        self.goal_reward = 1000

        self.start_state = START_STATE
        self.goal_state = GOAL_STATE

        # state space and action space (MlpPolicy needs these params for input)
        high_state = np.array([5., 2., 10., 2., np.pi, np.pi / 3])
        low_state = np.array([-5., -2., 0., -2., -np.pi, -np.pi / 3])

        high_obsrv = np.array([5., 2., 10., 2., np.pi, np.pi / 3] + [5*2] * self.num_lasers)
        low_obsrv = np.array([-5., -2., 0., -2., -np.pi, -np.pi/3] + [0] * self.num_lasers)

        # controls are two thrusts
        high_action = np.array([10., 10.])
        low_action = np.array([8., 8.])

        self.state_space = spaces.Box(low=low_state, high=high_state)
        self.observation_space = spaces.Box(low=low_obsrv, high=high_obsrv)
        self.action_space = spaces.Box(low=low_action, high=high_action)

        self.state_dim = 6
        self.action_dim = 2

        self.goal_pos_tolerance = 1.5
        self.goal_vel_limit = 0.25
        self.goal_phi_limit = np.pi / 6.
        self.pre_obsrv = None
        self.reward_type = None
        self.set_hover_end = None
        self.brsEngine = None

        self.goal_level_high = False

    def _discretize_laser(self, laser_data, new_ranges):

        discretized_ranges = []
        # mod = int(len(laser_data.ranges)/new_ranges)
        # for i, item in enumerate(laser_data.ranges):
        #     if (i+1) % mod == 0:
        #         if laser_data.ranges[i] == float('Inf') or np.isinf(laser_data.ranges[i]):
        #             discretized_ranges.append(10)
        #         elif np.isnan(laser_data.ranges[i]):
        #             discretized_ranges.append(0)
        #         else:
        #             discretized_ranges.append(int(laser_data.ranges[i]))

        full_ranges = float(len(laser_data.ranges))
        # print("laser ranges num: %d" % full_ranges)

        for i in range(new_ranges):
            new_i = int(i * full_ranges // new_ranges + full_ranges // (2 * new_ranges))
            if laser_data.ranges[new_i] == float('Inf') or np.isinf(laser_data.ranges[new_i]):
                discretized_ranges.append(10)
            elif np.isnan(laser_data.ranges[new_i]):
                discretized_ranges.append(0)
            else:
                discretized_ranges.append(int(laser_data.ranges[new_i]))

        return discretized_ranges

    def _in_obst(self, laser_data, dynamic_data):
        laser_min_range = 0.6
        # collision_min_range = 0.8
        tmp_x = dynamic_data.pose.position.x
        tmp_y = dynamic_data.pose.position.y
        tmp_z = dynamic_data.pose.position.z

        if tmp_z <= laser_min_range:
            print("tmp_z:", tmp_z)
            return True
        if Euclid_dis((tmp_x, tmp_z), OBSTACLES_POS[0]) < 1.3 or Euclid_dis((tmp_x, tmp_z), OBSTACLES_POS[1]) < 1.3 \
                or Euclid_dis((tmp_x, tmp_z), OBSTACLES_POS[2]) < 1.5 \
                or np.abs(tmp_x - WALLS_POS[0][0]) < laser_min_range or np.abs(tmp_x - WALLS_POS[1][0]) < laser_min_range \
                or np.abs(tmp_z - WALLS_POS[2][1]) < laser_min_range:
            return True

        # check the obstacle by sensor reflecting data
        # for idx, item in enumerate(laser_data.ranges):
        #     if laser_min_range > laser_data.ranges[idx] > 0:
        #         return True
        return False

    def _in_goal(self, state):

        assert len(state) == self.state_dim

        x = state[0]
        z = state[2]
        # print("z pos:", z)

        phi = state[4]
        # print("phi:", phi)

        # just consider pose restriction
        if self.set_hover_end == 'false':
            if np.sqrt((x - self.goal_state[0]) ** 2 + (z - self.goal_state[2]) ** 2) <= self.goal_pos_tolerance:
                return True
            else:
                return False
        elif self.set_hover_end == 'true':
            if np.sqrt((x - self.goal_state[0]) ** 2 + (z - self.goal_state[2]) ** 2) <= self.goal_pos_tolerance \
                    and (abs(phi - self.goal_state[4]) < 0.40):
                return True
            else:
                return False

            # two-levels goal reward setting
            # if np.sqrt((x - self.goal_state[0]) ** 2 + (z - self.goal_state[2]) ** 2) <= self.goal_pos_tolerance:
            #     if abs(phi - self.goal_state[4]) < 0.40:
            #         self.goal_level_high = True
            #     else:
            #         self.goal_level_high = False
            #     return True
            # else:
            #     return False
        else:
            raise ValueError("invalid param for set_hover_end!")

    def get_obsrv(self, laser_data, dynamic_data):

        discretized_laser_data = self._discretize_laser(laser_data, 8)

        # planar quadrotor x position
        x = dynamic_data.pose.position.x
        # planar quadrotor z position
        z = dynamic_data.pose.position.z

        # planar quadrotor velocity at x axis,
        vx = dynamic_data.twist.linear.x
        # planar quadrotor velocity at y axis == real world velocity z axis
        vz = dynamic_data.twist.linear.z

        ox = dynamic_data.pose.orientation.x
        oy = dynamic_data.pose.orientation.y
        oz = dynamic_data.pose.orientation.z
        ow = dynamic_data.pose.orientation.w

        # planar quadrotor pitch angle (along x-axis)
        _, pitch, _ = euler_from_quaternion([ox, oy, oz, ow])

        # planar quadrotor pitch angular velocity
        w = dynamic_data.twist.angular.y

        obsrv = [x, vx, z, vz, pitch, w] + discretized_laser_data

        return obsrv

    def reset(self):
        # Resets the state of the environment and returns an initial observation.
        rospy.wait_for_service('/gazebo/reset_simulation')
        try:
            self.reset_proxy()
            pose = Pose()
            pose.position.x = np.random.uniform(low=START_STATE[0] - 0.5, high=START_STATE[0] + 0.5)
            pose.position.y = self.get_model_state(model_name="quadrotor").pose.position.y
            pose.position.z = np.random.uniform(low=START_STATE[2] - 0.5, high=START_STATE[2] + 0.5)
            pitch = np.random.uniform(low=START_STATE[4] - 0.1, high=START_STATE[4] + 0.1)
            ox, oy, oz, ow = quaternion_from_euler(0.0, pitch, 0.0)
            pose.orientation.x = ox
            pose.orientation.y = oy
            pose.orientation.z = oz
            pose.orientation.w = ow

            reset_state = ModelState()
            reset_state.model_name = "quadrotor"
            reset_state.pose = pose
            self.set_model_state(reset_state)
        except rospy.ServiceException as e:
            print("# Resets the state of the environment and returns an initial observation.")

        # rospy.sleep(5.)

        # Unpause simulation to make observation
        rospy.wait_for_service('/gazebo/unpause_physics')
        try:
            self.unpause()
        except rospy.ServiceException as e:
            print("/gazebo/unpause_physics service call failed")

        # read laser data
        laser_data = None
        dynamic_data = None
        while laser_data is None and dynamic_data is None:
            try:
                laser_data = rospy.wait_for_message('/scan', LaserScan, timeout=5)
                # dynamic_data = rospy.wait_for_message('/gazebo/model_states', ModelStates, timeout=5)
                rospy.wait_for_service("/gazebo/get_model_state")
                try:
                    dynamic_data = self.get_model_state(model_name="quadrotor")
                except rospy.ServiceException as e:
                    print("/gazebo/unpause_physics service call failed")
            except:
                pass

        rospy.wait_for_service('/gazebo/pause_physics')
        try:
            self.pause()
        except rospy.ServiceException as e:
            print("/gazebo/pause_physics service call failed")

        obsrv = self.get_obsrv(laser_data, dynamic_data)
        self.pre_obsrv = obsrv

        return np.asarray(obsrv)

    def step(self, action):
        rospy.wait_for_service('/gazebo/unpause_physics')
        try:
            self.unpause()
        except rospy.ServiceException as e:
            print("/gazebo/unpause_physics service call failed")

        # clip action
        # print("original action output from NN:", action)
        # action = np.clip(action + 9.0, self.action_space.low, self.action_space.high)
        # print("action after being clipped", action)

        # action = action + 8.5
        action = action + 8.8  # a little more power for easier launch away from ground
        if sum(np.isnan(action)) > 0:
            raise ValueError("Passed in nan to step! Action: " + str(action))

        pre_phi = self.pre_obsrv[4]

        wrench = Wrench()
        wrench.force.x = (action[0] + action[1]) * np.sin(pre_phi)
        wrench.force.y = 0
        # wrench.force.z = action[0] + action[1]
        wrench.force.z = (action[0] + action[1]) * np.cos(pre_phi)
        wrench.torque.x = 0
        wrench.torque.y = (action[0] - action[1]) * 0.4
        # wrench.torque.y = 1.0
        wrench.torque.z = 0

        rospy.wait_for_service('/gazebo/apply_body_wrench')
        self.force(body_name="base_link", reference_frame="world", wrench=wrench, start_time=rospy.Time().now(), duration=rospy.Duration(1))

        # rospy.wait_for_service("/gazebo/apply_joint_effort")
        # try:
        #     status_left = self.apply_joint_effort(joint_name='left_joint', effort=action[0],
        #                                           start_time=rospy.Time().now(), duration=rospy.Duration(10))
        #     status_right = self.apply_joint_effort(joint_name='right_joint', effort=action[1],
        #                                            start_time=rospy.Time().now(), duration=rospy.Duration(10))
        #     rospy.loginfo("status left: %d, %s, %d" % (status_left.success, status_left.status_message, rospy.Time().now().to_sec()))
        #     print("status right:", status_right.success, status_right.status_message)
        # except rospy.ServiceException as e:
        #     print("/gazebo/apply_joint_effort service call failed")

        laser_data = None
        dynamic_data = None
        while laser_data is None and dynamic_data is None:
            try:
                laser_data = rospy.wait_for_message('/scan', LaserScan, timeout=5)
                # dynamic_data = rospy.wait_for_message('/gazebo/model_states', ModelStates)
                rospy.wait_for_service("/gazebo/get_model_state")
                try:
                    dynamic_data = self.get_model_state(model_name="quadrotor")
                except rospy.ServiceException as e:
                    print("/gazebo/unpause_physics service call failed")
            except:
                pass

        # self.clear_joint_effort(joint_name='left_joint')
        # self.clear_joint_effort(joint_name='right_joint')

        rospy.wait_for_service('/gazebo/pause_physics')
        try:
            # resp_pause = pause.call()
            self.pause()
        except rospy.ServiceException as e:
            print("/gazebo/pause_physics service call failed")

        obsrv = self.get_obsrv(laser_data, dynamic_data)
        self.pre_obsrv = obsrv

        assert self.reward_type is not None
        reward = 0

        if self.reward_type == 'hand_craft':
            # reward = -self.control_reward_coff * (action[0] ** 2 + action[1] ** 2)
            # reward = -1
            reward = 0
        elif self.reward_type == 'ttr' and self.brsEngine is not None:
            # Notice z-axis ttr space is defined from (-5,5), in gazebo it's in (0,10), so you need -5 when you want correct ttr reward
            ttr_obsrv = copy.deepcopy(obsrv)
            ttr_obsrv[2] = ttr_obsrv[2] - 5
            ttr = self.brsEngine.evaluate_ttr(np.reshape(ttr_obsrv[:6], (1, -1)))
            reward = -ttr
        elif self.reward_type == 'distance':
            reward = -(Euclid_dis((obsrv[0], obsrv[2]), (GOAL_STATE[0], GOAL_STATE[2])))

        done = False
        suc = False

        # 1. when collision happens, done = True
        if self._in_obst(laser_data, dynamic_data):
            reward += self.collision_reward
            done = True

        # 2. In the neighbor of goal state, done is True as well. Only considering velocity and pos
        if self._in_goal(np.array(obsrv[:6])):

            # two-levels goal reward setting
            # if self.goal_level_high:
            #     reward += self.goal_reward
            # else:
            #     reward += self.goal_reward / 2.
            # print("high level goal? ", self.goal_level_high)
            # self.goal_level_high = False

            reward += self.goal_reward
            done = True
            suc = True
            print("in goal reward:", reward)

            if abs(obsrv[4] - self.goal_state[4]) < 0.40:
                print("good tilting!")

        if obsrv[4] > 1.2 or obsrv[4] < -1.2:
            reward += self.collision_reward * 2
            done = True

        # 3. Maybe episode length limit is another factor for resetting the robot, stay tuned.
        # waiting to be implemented
        # if action[0] >= 10 or action[1] >= 10:
        #     reward += self.collision_reward
        #     done = True
        # ---
        # rospy.loginfo("reward: %f", reward)
        # assert reward < 0
        # print("reward,", reward)
        return np.asarray(obsrv), reward, done, suc, {}

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]
