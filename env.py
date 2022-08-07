
import gym
from gym import spaces
from gym.utils import seeding
import numpy as np
from os import path
import time
import control

def calc_dstate(state, u):
    #if y > 0:
    #else:
    pass


def normalize_state(state):
    #if y < 0:
    #   y = 0
    pass

class RabbitState():
    def __init__(self, randomize=False):
        #const
        self.dt=.05
        self.g = g
        self.m0 = 1.
        self.m1 = 1.
        self.m2 = 2.
        self.l0 = 1.
        self.l1 = 1.
        self.l2 = 1.

        #constraint
        self.max_torque=2.

        self.max_x0   = 10.
        self.max_y0   = 10.
        self.max_th0  = np.pi
        self.max_th1  = np.pi
        self.max_th2  = np.pi
        self.max_dx0  = 5.
        self.max_dy0  = 5.
        self.max_dth0 = 8.
        self.max_dth1 = 8.
        self.max_dth2 = 8.
        self.min_x0   = -10.
        self.min_y0   = -10.
        self.min_th0  = 0
        self.min_th1  = 0
        self.min_th2  = -np.pi
        self.min_dx0  = -5.
        self.min_dy0  = -5.
        self.min_dth0 = -8.
        self.min_dth1 = -8.
        self.min_dth2 = -8.

        #state
        self.x0   = 0.
        self.y0   = 0.
        self.th0  = np.pi/2
        self.th1  = np.pi/2
        self.th2  = -np.pi/3
        self.dx0  = 0.
        self.dy0  = 0.
        self.dth0 = 0.
        self.dth1 = 0.
        self.dth2 = 0.

        if randomize:
            self.th0 = self.np_random.uniform(low=-np.pi/4, high=self.np.pi/4)
            self.th1 = self.np_random.uniform(low=-np.pi/4, high=self.np.pi/4)
            self.th2 = self.np_random.uniform(low=-np.pi/4, high=self.np.pi/4)

    def obs(self, u):
        return np.array([ self.th0
                        , self.th1
                        , self.th2
                        , self.dth0
                        , self.dth1
                        , self.dth2
        ], dtype=np.float32)

        self.observation_space = spaces.Box(low=ob_low, high=ob_high, dtype=np.float32)
        max_action = np.array([self.max_torque, self.max_torque])
        self.action_space = spaces.Box(low=-max_action, high=max_action, dtype=np.float32)

    def ob_high(self):
        return np.array([ np.pi #th0
                        , np.pi #th1
                        , np.pi #th2
                        , self.max_rot_speed #dth0
                        , self.max_rot_speed #dth1
                        , self.max_rot_speed #dth2
        ], dtype=np.float32)

    def ob_low(self):
        return np.array([ 0      #th0
                        , 0      #th1
                        , -np.pi #th2
                        , self.max_rot_speed #dth0
                        , self.max_rot_speed #dth1
                        , self.max_rot_speed #dth2
        ], dtype=np.float32)


class RabbitEnv(gym.Env):
    metadata = {
        'render.modes' : ['human', 'rgb_array'],
        'video.frames_per_second' : 30
    }

    def __init__(self):
        self.state = RabbitEnv(False)
        self.viewer = None
        self.frame_no = 0

        self.observation_space = spaces.Box(low=ob_low, high=ob_high, dtype=np.float32)
        max_action = np.array([self.max_torque, self.max_torque])
        self.action_space = spaces.Box(low=-max_action, high=max_action, dtype=np.float32)

        self.seed()


    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]


    def step(self, u):
        th, thdot = self.state # th := theta

        g = self.g
        m = self.m
        l = self.l
        dt = self.dt

        u = np.clip(u, -self.max_torque, self.max_torque)

        costs = angle_normalize(th)**2 + .1*thdot**2 + .001*(u**2)

        dstate = calc_dstate(state, u)
        state = state + dstate * dt

        newthdot = thdot + (-3*g/(2*l) * np.sin(th + np.pi) + 3./(m*l**2)*u) * dt
        newth = normalize_state(th + newthdot*dt)

        self.state = newth
        self.frame_no = self.frame_no + 1

        #print((self.frame_no, angle_normalize(th), thdot))

        theta, thetadot = self.state
        obs = np.array([np.cos(theta), np.sin(theta), thetadot])

        return self._get_obs(), -costs, False, {}


    def reset(self):
        self.state = RabbitEnv(False)
        self.frame_no = 0
        return self.state.obs(u = np.array([0, 0]))


    def render(self, mode='human'):

        if self.viewer is None:
            from gym.envs.classic_control import rendering
            self.viewer = rendering.Viewer(500,500)
            self.viewer.set_bounds(-2.2,2.2,-2.2,2.2)
            rod = rendering.make_capsule(1, .2)
            rod.set_color(.8, .3, .3)
            self.pole_transform = rendering.Transform()
            rod.add_attr(self.pole_transform)
            self.viewer.add_geom(rod)
            axle = rendering.make_circle(.05)
            axle.set_color(0,0,0)
            self.viewer.add_geom(axle)

        self.pole_transform.set_rotation(self.state[0] + np.pi/2)
        #if self.frame_no < 3:
        #    time.sleep(1)
        #time.sleep(0.1)

        return self.viewer.render(return_rgb_array = mode=='rgb_array')

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None

def angle_normalize(x):
    return (((x+np.pi) % (2*np.pi)) - np.pi)
