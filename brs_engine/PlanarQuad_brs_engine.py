import matlab.engine
import os
import numpy as np
from scipy.interpolate import RectBivariateSpline, interp1d
from gym_foo.gym_foo.envs.PlanarQuadEnv_v0 import PlanarQuadEnv_v0

# Note: the state variables and dims could not be equal to that in learning process
# But now, after discussion with Mo, we decide to use the same state variables as learning, but different action variables.
# And we also decide to use system-decomposition

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

GOAL_STATE = np.array([4., 0., 4., 0., 0.75, 0.])
# START_STATE = np.array([-3.182, 0., 3., 0., 0., 0.])


class Quadrotor_brs_engine(object):
    # Starts and sets up the MATLAB engine that runs in the background.
    def __init__(self):
        self.eng = matlab.engine.start_matlab()
        

        cur_path = os.path.dirname(os.path.abspath(__file__))
        self.eng.workspace['cur_path'] = cur_path
        self.eng.workspace['home_path'] = os.environ['PROJ_HOME'] 
        self.eng.eval("addpath(genpath([home_path, '/toolboxls']));", nargout=0)
        self.eng.eval("addpath(genpath([home_path, '/helperOC']));", nargout=0)
        self.eng.eval("addpath(genpath(cur_path));", nargout=0)
    
    def reset_variables(self, tMax=15.0, interval=0.1, nPoints=41):

        self.state_dim = len(GOAL_STATE)
        self.Thrustmin = 0
        self.Thrustmax = 1.0 * 1.25 * 9.81

        self.goal_state = matlab.double([[GOAL_STATE[0]],[GOAL_STATE[1]],[GOAL_STATE[2]],[GOAL_STATE[3]],[GOAL_STATE[4]],[GOAL_STATE[5]]])
        self.goal_radius = matlab.double([[1.0],[0.5],[1.0],[0.5],[np.pi/3],[0.25]])

        self.gMin = matlab.double([[-5.], [-2.], [-5.], [-2.], [-np.pi], [-np.pi/2]])
        self.gMax = matlab.double([[5.], [2.], [5.], [2.], [np.pi], [np.pi/2]])
        self.nPoints = nPoints
        self.gN = matlab.double((self.nPoints * np.ones((self.state_dim, 1))).tolist())
        self.axis_coords = [np.linspace(self.gMin[i][0], self.gMax[i][0], nPoints) for i in range(self.state_dim)]

        # In quadrotor env, target region is set to rectangle, not cylinder
        self.goalRectAndState = matlab.double([[GOAL_STATE[0]-1],[GOAL_STATE[2]-1],[GOAL_STATE[0]+1],[GOAL_STATE[2]+1],
                                               [GOAL_STATE[5]], [GOAL_STATE[1]], [GOAL_STATE[3]], [GOAL_STATE[4]]])

        (self.initTargetAreaX, self.initTargetAreaY, self.initTargetAreaW, self.initTargetAreaVxPhi, self.initTargetAreaVyPhi) = \
            self.eng.Quad6D_create_init_target(self.gMin,
                                                self.gMax,
                                                self.gN,
                                                self.goalRectAndState,
                                                self.goal_radius,
                                                nargout=5)

        self.T1Min = self.T2Min = float(self.Thrustmin)
        self.T1Max = self.T2Max = float(self.Thrustmax)
        self.wRange = matlab.double([[-np.pi/2, np.pi/2]])
        self.vxRange = matlab.double([[-2., 2.]])
        self.vyRange = matlab.double([[-2., 2.]])

        self.tMax = float(tMax)
        self.interval = float(interval)

        # These functions' order should be correct
        cur_path = os.path.dirname(os.path.abspath(__file__))
        # cur_path = os.getcwd() + '/brs_engine'
        if os.path.exists(cur_path + '/ttrX.mat') and os.path.exists(cur_path + '/ttrY.mat') and os.path.exists(cur_path + '/ttrW.mat') \
            and os.path.exists(cur_path + '/ttrVxPhi.mat') and os.path.exists(cur_path + '/ttrVyPhi.mat'):
            self.eng.eval("load ttrX.mat ttrX", nargout=0)
            self.eng.eval("load ttrY.mat ttrY", nargout=0)
            self.eng.eval("load ttrW.mat ttrW", nargout=0)
            self.eng.eval("load ttrVxPhi.mat ttrVxPhi", nargout=0)
            self.eng.eval("load ttrVyPhi.mat ttrVyPhi", nargout=0)
            self.ttrX = self.eng.workspace['ttrX']
            self.ttrY = self.eng.workspace['ttrY']
            self.ttrW = self.eng.workspace['ttrW']
            self.ttrVxPhi = self.eng.workspace['ttrVxPhi']
            self.ttrVyPhi = self.eng.workspace['ttrVyPhi']
        else:
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
                                       self.interval,
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

        self.eng.workspace['ttrX'] = self.ttrX
        self.eng.workspace['ttrY'] = self.ttrY
        self.eng.workspace['ttrW'] = self.ttrW
        self.eng.workspace['ttrVxPhi'] = self.ttrVxPhi
        self.eng.workspace['ttrVyPhi'] = self.ttrVyPhi
        self.eng.eval("save ttrX.mat ttrX", nargout=0)
        self.eng.eval("save ttrY.mat ttrY", nargout=0)
        self.eng.eval("save ttrW.mat ttrW", nargout=0)
        self.eng.eval("save ttrVxPhi.mat ttrVxPhi", nargout=0)
        self.eng.eval("save ttrVyPhi.mat ttrVyPhi", nargout=0)

    def ttr_interpolation(self):
        np_tX = np.asarray(self.ttrX)[:, -1]
        np_tY = np.asarray(self.ttrY)[:, -1]
        np_tW = np.asarray(self.ttrW)[:, -1]
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
        self.x_ttr_check = interp1d(x=self.axis_coords[X_IDX], y=np_tX, fill_value='extrapolate')
        self.y_ttr_check = interp1d(x=self.axis_coords[Y_IDX], y=np_tY, fill_value='extrapolate')
        self.w_ttr_check = interp1d(x=self.axis_coords[W_IDX], y=np_tW, fill_value='extrapolate', kind='nearest')

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
        # self.eng.workspace['Z'] = self.ttr_value[idx]
        # # self.eng.workspace['theta_min'] = self.gMin[THETA_IDX][0]
        # # self.eng.workspace['theta_max'] = self.gMax[THETA_IDX][0]
        #
        #
        # # self.eng.eval("theta = linspace(theta_min, theta_max, nPoints);", nargout=0)
        # # # self.eng.eval("scatter3(X(:),Y(:),THETA(:),5,Z(:)，'filled');", nargout=0)
        # #
        #
        # # for idx in range(0, self.nPoints-1, 3):
        # #
        # #     self.eng.eval("surf(X,Y,Z,Z);", nargout=0)
        # #     self.eng.eval("hold on;", nargout=0)
        # # self.eng.workspace['tMax'] = float(self.tMax)
        # # self.eng.workspace['interval'] = float(self.interval)
        # # self.eng.workspace['ttr'] = self.ttr_value
        # # self.eng.eval("save ttr.mat ttr", nargout=0)

    def evaluate_ttr(self, states):
        # return (self.vxphi_ttr_check(states[:, VX_IDX], states[:, PHI_IDX], grid=False)
        #         + self.vyphi_ttr_check(states[:, VY_IDX], states[:, PHI_IDX], grid=False)
        #         + self.x_ttr_check(states[:, X_IDX])
        #         + self.y_ttr_check(states[:, Y_IDX])
        #         + self.w_ttr_check(states[:, W_IDX]))
        # print("state:", states)
        assert not np.isnan(self.vxphi_ttr_check(states[:, VX_IDX], states[:, PHI_IDX], grid=False))
        assert not np.isnan(self.vyphi_ttr_check(states[:, VY_IDX], states[:, PHI_IDX], grid=False))
        assert not np.isnan(self.x_ttr_check(states[:, X_IDX]))
        assert not np.isnan(self.y_ttr_check(states[:, Y_IDX]))
        assert not np.isnan(self.w_ttr_check(states[:, W_IDX]))
        tmp_ttr = (self.vxphi_ttr_check(states[:, VX_IDX], states[:, PHI_IDX], grid=False),
                    self.vyphi_ttr_check(states[:, VY_IDX], states[:, PHI_IDX], grid=False),
                    self.x_ttr_check(states[:, X_IDX]),
                    self.y_ttr_check(states[:, Y_IDX]),
                    self.w_ttr_check(states[:, W_IDX]))
        rslt = np.max(tmp_ttr, axis=0)
        return rslt[0]

if __name__ == "__main__":
    quad_engine = Quadrotor_brs_engine()
    quad_engine.reset_variables()


