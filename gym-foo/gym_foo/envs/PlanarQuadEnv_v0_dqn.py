import gym
from gym import spaces
from gym.utils import seeding
import numpy as np
import gazebo_env

from geometry_msgs.msg import Twist
from sensor_msgs.msg import LaserScan
from sensor_msgs.msg import JointState
from std_srvs.srv import Empty
from gazebo_msgs.srv import GetModelState
import rospy

from hector_uav_msgs.srv import EnableMotors

from tf.transformations import euler_from_quaternion

# need to be compatitable with model.sdf and world.sdf for custom setting
# notice: it's not the gazebo pose state, not --> x,y,z,pitch,roll,yaw !!
GOAL_STATE = np.array([4., 0., 4., 0., 0., 0.])
START_STATE = np.array([-3.182, 0., 3., 0., 0., 0.])


class PlanarQuadEnv_v0_dqn(gazebo_env.GazeboEnv):

    def __init__(self):
        # Launch the simulation with the given launchfile name
        gazebo_env.GazeboEnv.__init__(self, "QuadrotorAirSpace_v0.launch")
        self.vel_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=5)
        self.unpause = rospy.ServiceProxy('/gazebo/unpause_physics', Empty)
        self.pause = rospy.ServiceProxy('/gazebo/pause_physics', Empty)
        self.reset_proxy = rospy.ServiceProxy('/gazebo/reset_simulation', Empty)
        self.get_model_states = rospy.ServiceProxy('/gazebo/get_model_state', GetModelState)
        self.enable_moters = rospy.ServiceProxy('/enable_motors', EnableMotors)

        # First you need enable motor and do anything else
        rospy.wait_for_service('/enable_motors')
        try:
            enable_motors = rospy.ServiceProxy('/enable_motors', EnableMotors)
            res = enable_motors(True)
            if res:
                print("Motors enabled!")
            else:
                print("Failed to enable motors...")
        except rospy.ServiceException as e:
            print("Enable service", '/enable_motors', "call failed: %s" % e)

        self._seed()

        self.m = 1.25
        self.g = 9.81
        self.laser_num = 100

        self.Thrustmax = 1.00 * self.m * self.g
        self.Thrustmin = 0

        self.control_reward_coff = 0.01
        self.collision_reward = -2 * 200 * self.control_reward_coff * (self.Thrustmax ** 2)
        self.goal_reward = 1000

        self.start_state = START_STATE
        self.goal_state = GOAL_STATE

        # state space and action space (MlpPolicy needs these params for input)
        high_state = np.array([5., 2., 5., 2., np.pi, np.pi / 6])
        high_obsrv = np.array([5., 2., 5., 2., np.pi, np.pi / 6] + [5*2] * self.laser_num)
        # controls are vx and vz
        high_action = np.array([3, 3])

        self.state_space = spaces.Box(low=-high_state, high=high_state)
        self.observation_space = spaces.Box(low=np.concatenate((-high_state, np.array([0]*self.laser_num)), axis=0), high=high_obsrv)
        self.action_space = spaces.Box(low=-high_action, high=high_action)

        self.state_dim = 6
        self.action_dim = 2

        self.goal_pos_tolerance = 1.0

        self.pre_obsrv = None
        self.reward_type = None
        self.brsEngine = None

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
        print("laser ranges num: %d" % full_ranges)

        for i in range(new_ranges):
            new_i = int(i * full_ranges // new_ranges + full_ranges // (2 * new_ranges))
            if laser_data.ranges[new_i] == float('Inf') or np.isinf(laser_data.ranges[new_i]):
                discretized_ranges.append(10)
            elif np.isnan(laser_data.ranges[new_i]):
                discretized_ranges.append(0)
            else:
                discretized_ranges.append(int(laser_data.ranges[new_i]))

        return discretized_ranges

    def _in_obst(self, laser_data):

        min_range = 0.2
        for idx, item in enumerate(laser_data.ranges):
            if min_range > laser_data.ranges[idx] > 0:
                return True
        return False

    def _in_goal(self, state):

        assert len(state) == self.state_dim

        x = state[0]
        y = state[2]

        # just consider pose restriction
        if np.sqrt((x - self.goal_state[0]) ** 2 + (y - self.goal_state[2]) ** 2) <= self.goal_pos_tolerance:
            return True
        else:
            return False

    def get_obsrv(self, laser_data, dynamic_data):

        # discretized_laser_data = self._discretize_laser(laser_data, 8)
        discretized_laser_data = self._discretize_laser(laser_data, self.laser_num)

        # planar quadrotor x position
        x = dynamic_data.pose.position.x
        # planar quadrotor y position == real world z position
        y = dynamic_data.pose.position.z - 5

        # planar quadrotor velocity at x axis,
        vx = dynamic_data.twist.linear.x
        # planar quadrotor velocity at y axis == real world velocity z axis
        vy = dynamic_data.twist.linear.z

        ox = dynamic_data.pose.orientation.x
        oy = dynamic_data.pose.orientation.y
        oz = dynamic_data.pose.orientation.z
        ow = dynamic_data.pose.orientation.w

        # planar quadrotor rolling angle
        roll, _, _ = euler_from_quaternion([ox, oy, oz, ow])

        # planar quadrotor rolling angular velocity
        w = dynamic_data.twist.angular.y

        obsrv = [x, vx, y, vy, roll, w] + discretized_laser_data

        print("observation length: %d" % len(obsrv))

        return obsrv

    # def map_action(self, action):
    #     # need this because our action space is [-3,3] for two thrusts
    #     return [self.Thrustmin + (0.5 + a/6.0)*(self.Thrustmax - self.Thrustmin) for a in action]

    def reset(self):
        # Resets the state of the environment and returns an initial observation.
        rospy.wait_for_service('/gazebo/reset_simulation')
        try:
            self.reset_proxy()
        except rospy.ServiceException as e:
            print("# Resets the state of the environment and returns an initial observation.")

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
                    dynamic_data = self.get_model_states(model_name="quadrotor")
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

        # first you need to map action to positive value. action = [T1, T2]
        # action = self.map_action(action)

        if np.isnan(action):
            raise ValueError("Passed in nan to step! Action: " + str(action))

        # clip action
        # action = np.clip(action, self.action_space.low, self.action_space.high)

        cmd_vel = Twist()

        max_linear_speed = 0.8

        cmd_vel.linear.x = (action - 10) * max_linear_speed * 0.1  # from (-0.8 to + 0.8)
        cmd_vel.linear.y = 0
        cmd_vel.linear.z = 0.2

        # cmd_vel.angular.x = 0.1
        # cmd_vel.angular.z = 0
        # cmd_vel.angular.y = 3

        self.vel_pub.publish(cmd_vel)

        laser_data = None
        dynamic_data = None
        while laser_data is None and dynamic_data is None:
            try:
                laser_data = rospy.wait_for_message('/scan', LaserScan, timeout=5)
                # dynamic_data = rospy.wait_for_message('/gazebo/model_states', ModelStates)
                rospy.wait_for_service("/gazebo/get_model_state")
                try:
                    dynamic_data = self.get_model_states(model_name="quadrotor")
                except rospy.ServiceException as e:
                    print("/gazebo/unpause_physics service call failed")
            except:
                pass

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
            reward = -1
        elif self.reward_type == 'ttr' and self.brsEngine is not None:
            ttr = self.brsEngine.evaluate_ttr(np.reshape(obsrv[:6], (1, -1)))
            reward = -ttr

        done = False
        # suc = False

        # 1. when collision happens, done = True
        if self._in_obst(laser_data):
            reward += self.collision_reward
            done = True

        # 2. In the neighbor of goal state, done is True as well. Only considering velocity and pos
        if self._in_goal(np.array(obsrv[:6])):
            reward += self.goal_reward
            done = True
            # suc = True

        # 3. Maybe episode length limit is another factor for resetting the robot, stay tuned.
        # waiting to be implemented
        # ---

        return np.asarray(obsrv), reward, done, {}

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]
