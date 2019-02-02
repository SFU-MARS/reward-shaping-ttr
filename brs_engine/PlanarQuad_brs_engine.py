import matlab.engine
import os
import numpy as np
from scipy.interpolate import RectBivariateSpline, interp1d
from gym_foo.envs.PlanarQuadEnv_v0 import PlanarQuadEnv_v0
# state variables index
X_IDX = 0
VX_IDX = 1
Y_IDX = 2
VY_IDX = 3
PHI_IDX = 4
W_IDX = 5

# action variables index
T1_IDX = 0
T2_IDX = 1


class Quadrotor_brs_engine:
    # Starts and sets up the MATLAB engine that runs in the background.
    def __init__(self):
        self.eng = matlab.engine.start_matlab()
        self.eng.cd("/home/xlv/Desktop/IROS2019/brs_engine", nargout=0)
        self.eng.eval("addpath(genpath('/home/xlv/Desktop/toolboxls/Kernel'))", nargout=0)
        self.eng.eval("addpath(genpath('/home/xlv/Desktop/helperOC'));", nargout=0)

        self.env = PlanarQuadEnv_v0()

    def reset_variables(self, tMax=1, interval=0.001, nPoints=251.):

        goal_x, goal_vx, goal_y, goal_vy, goal_phi, goal_w = self.env.goal_state
        x_low, vx_low, y_low, vy_low, phi_low, w_low = self.env.state_space.low
        x_high, vx_high, y_high, vy_high, phi_high, w_high = self.env.state_space.high

        self.goal_state = matlab.double([[goal_x],[goal_vx],[goal_y],[goal_vy],[goal_phi],[goal_w]])
        self.goal_radius = matlab.double([[0.1],[0.25],[0.1],[0.25],[np.pi/6],[0.25]])

        self.gMin = matlab.double([[x_low], [vx_low], [y_low], [vy_low], [phi_low], [w_low]])
        self.gMax = matlab.double([[x_high],[vx_high],[y_high],[vy_high],[phi_high], [w_high]])
        self.nPoints = nPoints
        self.gN = matlab.double((self.nPoints * np.ones((self.env.state_dims, 1))).tolist())
        self.axis_coords = [np.linspace(self.gMin[i][0], self.gMax[i][0], nPoints) for i in range(self.env.state_dims)]

        # In quadrotor env, target region is set to rectangle, not cylinder
        self.goalRectAndState = matlab.double([[goal_x-1],[goal_y-1],[goal_x+1],[goal_y+1],
                                               [goal_w], [goal_vx], [goal_vy], [goal_phi]])

        (self.initTargetAreaX, self.initTargetAreaY, self.initTargetAreaW, self.initTargetAreaVxPhi, self.initTargetAreaVyPhi) = \
            self.eng.Quad6D_create_init_target(self.gMin,
                                                self.gMax,
                                                self.gN,
                                                self.goalRectAndState,
                                                self.goal_radius,
                                                nargout=5)

        self.T1Min = self.T2Min = float(self.env.Thrustmin)
        self.T1Max = self.T2Max = float(self.env.Thrustmax)
        self.wRange = matlab.double([[self.env.state_space.low[W_IDX]], [self.env.state_space.high[W_IDX]]]);
        self.vxRange = matlab.double([[self.env.state_space.low[VX_IDX]], [self.env.state_space.high[VX_IDX]]]);
        self.vyRange = matlab.double([[self.env.state_space.low[VY_IDX]], [self.env.state_space.high[VY_IDX]]]);

        self.tMax = float(tMax)
        self.interval = float(interval)

        # These functions' order should be correct
        self.get_value_function()
        self.get_ttr_function()
        self.ttr_interpolation()

    def get_value_function(self):
        (self.valueX, self.valueY, self.valueW, self.valueVxPhi, self.valueVyPhi) = \
            self.eng.Quad6D_approx_RS(self.gMin,
                                       self.gMax,
                                       self.gN,
                                       self.T1Min,
                                       self.T1Max,
                                       self.T2Min,
                                       self.T2Max,
                                       self.wRange,
                                       self.vxRange,
                                       self.vyRange,
                                       self.initTargetAreaX,
                                       self.initTargetAreaY,
                                       self.initTargetAreaW,
                                       self.initTargetAreaVxPhi,
                                       self.initTargetAreaVyPhi,
                                       self.tMax,
                                       nargout=5)

    def get_ttr_function(self):
        (self.ttrX, self.ttrY, self.ttrW, self.ttrVxPhi, self.ttrVyPhi) = \
            self.eng.Quad6D_approx_TTR(self.gMin,
                                        self.gMax,
                                        self.gN,
                                        self.valueX,
                                        self.valueY,
                                        self.valueW,
                                        self.valueVxPhi,
                                        self.valueVyPhi,
                                        self.tMax,
                                        self.interval,
                                        nargout=5)

    def ttr_interpolation(self):
        np_tX = np.asarray(self.ttrX)
        np_tY = np.asarray(self.ttrY)
        np_tW = np.asarray(self.ttrW)
        np_tVxPhi = np.asarray(self.ttrVxPhi)
        np_tVyPhi = np.asarray(self.ttrVyPhi)

        print('np_tX shape is', np_tX.shape, flush=True)
        print('np_tY shape is', np_tY.shape, flush=True)
        print('np_tW shape is', np_tW.shape, flush=True)
        print('np_tVxPhi shape is', np_tVxPhi.shape, flush=True)
        print('np_tVyPhi shape is', np_tVyPhi.shape, flush=True)

        # Here we interpolate based on discrete ttr function
        self.vxphi_ttr_check = RectBivariateSpline(x=self.axis_coords[VX_IDX],
                                            y=self.axis_coords[PHI_IDX],
                                            z=np_tVxPhi,
                                            kx=1, ky=1)
        self.vyphi_ttr_check = RectBivariateSpline(x=self.axis_coords[VY_IDX],
                                            y=self.axis_coords[PHI_IDX],
                                            z=np_tVyPhi,
                                            kx=1, ky=1)
        self.x_ttr_check = interp1d(x=self.axis_coords[X_IDX], y=np_tX)
        self.y_ttr_check = interp1d(x=self.axis_coords[Y_IDX], y=np_tY)
        self.w_ttr_check = interp1d(x=self.axis_coords[W_IDX], y=np_tW)

    def evaluate_ttr(self, states):
        # return (self.vxphi_ttr_check(states[:, VX_IDX], states[:, PHI_IDX], grid=False)
        #         + self.vyphi_ttr_check(states[:, VY_IDX], states[:, PHI_IDX], grid=False)
        #         + self.x_ttr_check(states[:, X_IDX])
        #         + self.y_ttr_check(states[:, Y_IDX])
        #         + self.w_ttr_check(states[:, W_IDX]))

        tmp_ttr = (self.vxphi_ttr_check(states[:, VX_IDX], states[:, PHI_IDX], grid=False),
                    self.vyphi_ttr_check(states[:, VY_IDX], states[:, PHI_IDX], grid=False),
                    self.x_ttr_check(states[:, X_IDX]),
                    self.y_ttr_check(states[:, Y_IDX]),
                    self.w_ttr_check(states[:, W_IDX]))
        rslt = np.max(tmp_ttr, axis=0)
        return rslt