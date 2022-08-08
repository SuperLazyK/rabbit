
import gym
from gym import spaces
from gym.utils import seeding
import numpy as np
from os import path
import time
import control

IDX_x    = 0
IDX_y    = 1
IDX_th0  = 2
IDX_th1  = 3
IDX_th2  = 4
IDX_dx   = 5
IDX_dy   = 6
IDX_dth0 = 7
IDX_dth1 = 8
IDX_dth2 = 9

def rhs_foot_contact(t, s, u, params):
    g  = params.get('g',     9.8)
    m0 = params.get('m0',    1.)
    m1 = params.get('m1',    1.)
    m2 = params.get('m2',    2.)
    l0 = params.get('l0',    1.)
    l1 = params.get('l1',    1.)
    l2 = params.get('l2',    1.)
    pass


def rhs_foot_in_the_air(t, s, u, params):
    g  = params.get('g',     9.8)
    m0 = params.get('m0',    1.)
    m1 = params.get('m1',    1.)
    m2 = params.get('m2',    2.)
    l0 = params.get('l0',    1.)
    l1 = params.get('l1',    1.)
    l2 = params.get('l2',    1.)
    pass


def rhs(t, s, u, params):
    assert s.y >= 0, ""
    if s[IDX_y] == 0:
        return rhs_foot_contact(t, s, u, params)
    else:
        return rhs_foot_in_the_air(t, s, u, params)


def obs(t, s, u, params):
    pass


model = ct.NonlinearIOSystem(rhs, obs
        , inputs=('tau1', 'tau2')
        , outputs=('th0', 'th1', 'th2', 'dth0', 'dth1', 'dth2', 'fy')
        , states=('x', 'y', 'th0', 'th1', 'th2', 'dx', 'dy', 'dth0', 'dth1', 'dth2')
        , name='rabit')


def step(x, u, p, te=1):
    ts=0
    p0 = {'F0':p}
    u = np.repeat(np.array(u).reshape(Nu,1), te-ts+1, axis=1)
    T = np.array(range(ts,te+1))
    t, x = ct.input_output_response(io_ex1_11, T, U=u, X0=x, params=p0)
    return x[:,-1]

#class RabbitState():
#    def __init__(self, randomize=False):
#        #const
#        self.dt=.05
#
#        #constraint
#        self.max_torque=2.
#
#        self.max_x0   = 10.
#        self.max_y0   = 10.
#        self.max_th0  = np.pi
#        self.max_th1  = np.pi
#        self.max_th2  = np.pi
#        self.max_dx0  = 5.
#        self.max_dy0  = 5.
#        self.max_dth0 = 8.
#        self.max_dth1 = 8.
#        self.max_dth2 = 8.
#        self.min_x0   = -10.
#        self.min_y0   = -10.
#        self.min_th0  = 0
#        self.min_th1  = 0
#        self.min_th2  = -np.pi
#        self.min_dx0  = -5.
#        self.min_dy0  = -5.
#        self.min_dth0 = -8.
#        self.min_dth1 = -8.
#        self.min_dth2 = -8.
#
#        #state
#        self.x0   = 0.
#        self.y0   = 0.
#        self.th0  = np.pi/2
#        self.th1  = np.pi/2
#        self.th2  = -np.pi/3
#        self.dx0  = 0.
#        self.dy0  = 0.
#        self.dth0 = 0.
#        self.dth1 = 0.
#        self.dth2 = 0.
#
#        if randomize:
#            self.th0 = self.np_random.uniform(low=-np.pi/4, high=self.np.pi/4)
#            self.th1 = self.np_random.uniform(low=-np.pi/4, high=self.np.pi/4)
#            self.th2 = self.np_random.uniform(low=-np.pi/4, high=self.np.pi/4)
#
#        self.model = ct.NonlinearIOSystem(self.rhs
#                , None
#                , inputs=('tau1', 'tau2')
#                , outputs=('c', 'T', 'h')
#                , states=('c', 'T', 'h')
#                , name='rabbit')
#
#    def obs_space(self):
#        ob_low = np.array([ self.min_th0
#                          , self.min_th1
#                          , self.min_th2
#                          , self.min_dth0
#                          , self.min_dth1
#                          , self.min_dth2
#                          , 0  # fy
#        ], dtype=np.float32)
#
#        ob_high = np.array([ self.max_th0
#                           , self.max_th1
#                           , self.max_th2
#                           , self.max_dth0
#                           , self.max_dth1
#                           , self.max_dth2
#                           , 10
#                    ], dtype=np.float32)
#
#        return spaces.Box(low=ob_low, high=ob_high, dtype=np.float32)
#
#
#    def act_space(self):
#        max_action = np.array([self.max_torque, self.max_torque])
#        return spaces.Box(low=-max_action, high=max_action, dtype=np.float32)
#
#
#    def step(self, u):
#        if self.y0 > 0:
#            self.step_in_
#
#        th, thdot = self.state # th := theta
#
#        g = self.g
#        m = self.m
#        l = self.l
#        dt = self.dt
#
#        u = np.clip(u, -self.max_torque, self.max_torque)
#        dstate = calc_dstate(state, u)
#        state = state + dstate * dt
#
#        newthdot = thdot + (-3*g/(2*l) * np.sin(th + np.pi) + 3./(m*l**2)*u) * dt
#        newth = normalize_state(th + newthdot*dt)



#class RabbitEnv(gym.Env):
#    metadata = {
#        'render.modes' : ['human', 'rgb_array'],
#        'video.frames_per_second' : 30
#    }
#
#    def __init__(self):
#        self.state = RabbitEnv(False)
#        self.viewer = None
#        self.frame_no = 0
#        self.observation_space = self.state.obs_space()
#        self.action_space = self.state.act_space()
#        self.seed()
#
#
#    def seed(self, seed=None):
#        self.np_random, seed = seeding.np_random(seed)
#        return [seed]
#
#    def reward(self):
#        #TODO
#        return 0
#
#    def check_done(self):
#        #TODO
#        return False
#
#    def step(self, u):
#        obs = self.state.step(u)
#        self.frame_no = self.frame_no + 1
#        return obs, self.reward(), self.check_done(), {}
#
#    def reset(self):
#        self.state = RabbitEnv(False)
#        self.frame_no = 0
#        obs = self.state.step(u = np.array([0, 0]))
#        return obs
#
#
#    def render(self, mode='human'):
#        #TODO
#
#        if self.viewer is None:
#            from gym.envs.classic_control import rendering
#            self.viewer = rendering.Viewer(500,500)
#            self.viewer.set_bounds(-2.2,2.2,-2.2,2.2)
#            rod = rendering.make_capsule(1, .2)
#            rod.set_color(.8, .3, .3)
#            self.pole_transform = rendering.Transform()
#            rod.add_attr(self.pole_transform)
#            self.viewer.add_geom(rod)
#            axle = rendering.make_circle(.05)
#            axle.set_color(0,0,0)
#            self.viewer.add_geom(axle)
#
#        self.pole_transform.set_rotation(self.state[0] + np.pi/2)
#        #if self.frame_no < 3:
#        #    time.sleep(1)
#        #time.sleep(0.1)
#
#        return self.viewer.render(return_rgb_array = mode=='rgb_array')
#
#    def close(self):
#        if self.viewer:
#            self.viewer.close()
#            self.viewer = None
#
