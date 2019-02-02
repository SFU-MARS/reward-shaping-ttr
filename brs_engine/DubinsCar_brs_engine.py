import matlab.engine
import os
import numpy as np
from scipy.interpolate import RectBivariateSpline, interp1d, RegularGridInterpolator
from gym_foo.envs.DubinsCarEnv_v0 import DubinsCarEnv_v0

# This import has side effects required for the kwarg projection='3d' in the call to fig.add_subplot
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

import pickle

# Note: the state variables and dims are not equal to that in learning process
# Specifically, here we use 3D state(xpos, ypos, theta) and 1D action(angular_velocity -- w) to compute TTR reward map for guided policy search

# state variables index
X_IDX = 0
Y_IDX = 1
THETA_IDX = 2
# V_IDX = 3
# W_IDX = 4

# action variables index
W_IDX = 0
# ACCEL_IDX = 0
# KAPPA_IDX = 1

GOAL_STATE = np.array([3.459, 3.626, 0.])
START_STATE = np.array([-0.182, -3.339, 0.])


class DubinsCar_brs_engine(object):
    # Starts and sets up the MATLAB engine that runs in the background.
    def __init__(self):
        self.eng = matlab.engine.start_matlab()
        self.eng.cd("/home/xlv/Desktop/IROS2019/brs_engine", nargout=0)
        self.eng.eval("addpath(genpath('/home/xlv/Desktop/toolboxls/Kernel'))", nargout=0)
        self.eng.eval("addpath(genpath('/home/xlv/Desktop/helperOC'));", nargout=0)

    def reset_variables(self, tMax=12.0, interval=0.1, nPoints=41):

        self.eng.eval("global gXYT;", nargout=0)

        self.state_dim = len(GOAL_STATE)

        self.goal_pos_tolerance = 1.0  # How many m away can you be from the goal and still finish?
        self.goal_theta_tolerance = 1.0  # How many radians different can you be from the goal theta and still finish?
        # self.goal_vel_tolerance = 0.2  # How many m/s away can you be from the goal and still finish?

        self.goal_state = matlab.double([[GOAL_STATE[0]], [GOAL_STATE[1]], [GOAL_STATE[2]]])
        self.goal_radius = matlab.double([[self.goal_pos_tolerance],
                                          [self.goal_pos_tolerance],
                                          [self.goal_theta_tolerance]])

        self.gMin = matlab.double([[-5.0], [-5.0], [-np.pi]])
        self.gMax = matlab.double([[5.0], [5.0], [np.pi]])
        self.nPoints = nPoints
        self.gN = matlab.double((self.nPoints * np.ones((self.state_dim, 1))).tolist())
        self.axis_coords = [np.linspace(self.gMin[i][0], self.gMax[i][0], nPoints) for i in range(self.state_dim)]

        self.initTargetArea = \
            self.eng.DubinsCar_create_init_target(self.gMin,
                                                  self.gMax,
                                                  self.gN,
                                                  self.goal_state,
                                                  self.goal_radius,
                                                  nargout=1)

        self.wMax = float(1.0)
        self.speed = float(1.0)
        self.tMax = float(tMax)
        self.interval = float(interval)

        # These functions' calling order should be correct
        self.get_value_function()
        self.get_ttr_function()
        self.ttr_interpolation()

        # print("tMax:%f\n" % tMax)
        # print("interval:%f\n" % interval)
        #
        # self.eng.eval("g = gXYT;", nargout=0)
        # self.eng.workspace['data'] = self.initTargetArea
        # self.eng.eval("visSetIm(g, data);", nargout=0)
        # self.eng.eval("hold on;", nargout=0)
        # self.eng.workspace['data'] = self.value
        # self.eng.eval("visSetIm(g, data);", nargout=0)
        #
        # self.eng.eval("savefig(strcat('ttr_dist_',num2str(tMax),'_', num2str(interval),'.fig')); hold on;", nargout=0)

    def get_value_function(self):
        self.value = \
            self.eng.DubinsCar_approx_RS(self.gMin,
                                         self.gMax,
                                         self.gN,
                                         self.wMax,
                                         self.speed,
                                         self.tMax,
                                         self.interval,
                                         self.initTargetArea,
                                         nargout=1)

    def get_ttr_function(self):
        self.ttr_value = \
            self.eng.DubinsCar_approx_TTR(self.gMin,
                                          self.gMax,
                                          self.gN,
                                          self.value,
                                          self.tMax,
                                          self.interval,
                                          nargout=1)

    def ttr_interpolation(self):

        np_ttr = np.asarray(self.ttr_value)
        print('np_ttr shape is', np_ttr.shape, flush=True)

        # Here we interpolate based on discrete ttr function
        # RegularGridInterpolator((x, y, z), data
        self.ttr_check = RegularGridInterpolator((self.axis_coords[X_IDX], self.axis_coords[Y_IDX], self.axis_coords[THETA_IDX]), np_ttr)

        # save the image of TTR function via matplotlib
        # fig = plt.figure()
        # ax = fig.add_subplot(111, projection='3d')
        # x = self.axis_coords[X_IDX]
        # y = self.axis_coords[THETA_IDX]
        # z = np_tXT
        # X, Y = np.meshgrid(x, y)
        # Z = z.reshape(X.shape)
        #
        # ax.plot_surface(X,Y,Z)
        # ax.set_xlabel('X Label')
        # ax.set_ylabel('Y Label')
        # ax.set_zlabel('Z Label')
        #
        # pickle.dump(fig, open('ttr_dist_'+str(self.tMax)+'_'+str(self.interval)+'.pickle', 'wb'))

        # save the image of TTR function via Matlab
        # self.eng.eval("fig = figure;", nargout=0)
        # self.eng.workspace['x_min'] = self.gMin[X_IDX][0]
        # self.eng.workspace['x_max'] = self.gMax[X_IDX][0]
        # self.eng.workspace['y_min'] = self.gMin[Y_IDX][0]
        # self.eng.workspace['y_max'] = self.gMax[Y_IDX][0]
        # self.eng.workspace['nPoints'] = float(self.nPoints)
        # self.eng.eval("x = linspace(x_min, x_max, nPoints);", nargout=0)
        # self.eng.eval("y = linspace(y_min, y_max, nPoints);", nargout=0)
        # self.eng.eval("[X,Y] = ndgrid(x, y);", nargout=0)
        # for idx in range(self.nPoints-1):
        #     self.eng.workspace['Z'] = self.ttr_value[idx]
        #     self.eng.eval("surf(X,Y,Z);", nargout=0)
        #     self.eng.eval("hold on;", nargout=0)
        # self.eng.workspace['tMax'] = float(self.tMax)
        # self.eng.workspace['interval'] = float(self.interval)

    def evaluate_ttr(self, states):
        return self.ttr_check(states[:, :])