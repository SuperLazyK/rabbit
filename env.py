
import gym
from gym import spaces
from gym.utils import seeding
import numpy as np
from os import path
import time
import control as ct
import sys

IDX_x0   = 0
IDX_y0   = 1
IDX_th0  = 2
IDX_th1  = 3
IDX_th2  = 4
IDX_dx   = 5
IDX_dy   = 6
IDX_dth0 = 7
IDX_dth1 = 8
IDX_dth2 = 9
Nu = 2
dt = 0.01
RENDER_OFFSET_Y = -2

MAX_TORQUE=2.

l0 =  1.
l1 =  1.
l2 =  1.

def rhs(t, s, u, params):
    g  = params.get('g',     9.8)
    m0 = params.get('m0',    1.)
    m1 = params.get('m1',    1.)
    m2 = params.get('m2',    2.)
    x0   = s[IDX_x0]
    y0   = s[IDX_y0]
    th0  = s[IDX_th0]
    th1  = s[IDX_th1]
    th2  = s[IDX_th2]
    dx   = s[IDX_dx  ]
    dy   = s[IDX_dy  ]
    dth0 = s[IDX_dth0]
    dth1 = s[IDX_dth1]
    dth2 = s[IDX_dth2]
    c0   = np.cos(th0)
    s0   = np.sin(th0)
    c1   = np.cos(th1)
    s1   = np.sin(th1)
    c2   = np.cos(th2)
    s2   = np.sin(th2)
    c01  = np.cos(th0 + th1)
    s01  = np.sin(th0 + th1)
    c12  = np.cos(th1 + th2)
    s12  = np.sin(th1 + th2)
    c012 = np.cos(th0 + th1 + th2)
    s012 = np.sin(th0 + th1 + th2)
    tau0 = 0
    tau1 = u[0]
    tau2 = u[1]

    A = np.array([
                [                                            1.0*m0 + 1.0*m1 + 1.0*m2,                                                                   0,                                                      -1.0*l0*m0*s0 - 1.0*m1*(l0*s0 + l1*s01) - 1.0*m2*(l0*s0 + l1*s01 + l2*s012),                                                -1.0*l1*m1*s01 - 1.0*m2*(l1*s01 + l2*s012),             -1.0*l2*m2*s012],
                [                                                                   0,                                                        m0 + m1 + m2,                                                                   c0*l0*m0 + m1*(c0*l0 + c01*l1) + m2*(c0*l0 + c01*l1 + c012*l2),                                                         c01*l1*m1 + m2*(c01*l1 + c012*l2),                  c012*l2*m2],
                [-l0*m0*s0 - l0*m1*s0 - l0*m2*s0 - l1*m1*s01 - l1*m2*s01 - l2*m2*s012, c0*l0*m0 + c0*l0*m1 + c0*l0*m2 + c01*l1*m1 + c01*l1*m2 + c012*l2*m2, 2*c1*l0*l1*m1 + 2*c1*l0*l1*m2 + 2*c12*l0*l2*m2 + 2*c2*l1*l2*m2 + l0**2*m0 + l0**2*m1 + l0**2*m2 + l1**2*m1 + l1**2*m2 + l2**2*m2, c1*l0*l1*m1 + c1*l0*l1*m2 + c12*l0*l2*m2 + 2*c2*l1*l2*m2 + l1**2*m1 + l1**2*m2 + l2**2*m2, l2*m2*(c12*l0 + c2*l1 + l2)],
                [                     -1.0*l1*m1*s01 - 1.0*l1*m2*s01 - 1.0*l2*m2*s012,                      1.0*c01*l1*m1 + 1.0*c01*l1*m2 + 1.0*c012*l2*m2,              1.0*c1*l0*l1*m1 + 1.0*c1*l0*l1*m2 + 1.0*c12*l0*l2*m2 + 2.0*c2*l1*l2*m2 + 1.0*l1**2*m1 + 1.0*l1**2*m2 + 1.0*l2**2*m2,                              2.0*c2*l1*l2*m2 + 1.0*l1**2*m1 + 1.0*l1**2*m2 + 1.0*l2**2*m2,      1.0*l2*m2*(c2*l1 + l2)],
                [                                                     -1.0*l2*m2*s012,                                                      1.0*c012*l2*m2,                                                                                                  1.0*l2*m2*(c12*l0 + c2*l1 + l2),                                                                    1.0*l2*m2*(c2*l1 + l2),                1.0*l2**2*m2]
            ])

    b = np.array([
                [  -1.0*c0*l0*m0*dth0**2 - 1.0*c0*l0*m1*dth0**2 - 1.0*c0*l0*m2*dth0**2 - 1.0*c01*l1*m1*dth0**2 - 2.0*c01*l1*m1*dth0*dth1 - 1.0*c01*l1*m1*dth1**2 - 1.0*c01*l1*m2*dth0**2 - 2.0*c01*l1*m2*dth0*dth1 - 1.0*c01*l1*m2*dth1**2 - 1.0*c012*l2*m2*dth0**2 - 2.0*c012*l2*m2*dth0*dth1 - 2.0*c012*l2*m2*dth0*dth2 - 1.0*c012*l2*m2*dth1**2 - 2.0*c012*l2*m2*dth1*dth2 - 1.0*c012*l2*m2*dth2**2],      # -fx0
                [                                   g*m0 + g*m1 + g*m2 - l0*m0*s0*dth0**2 - l0*m1*s0*dth0**2 - l0*m2*s0*dth0**2 - l1*m1*s01*dth0**2 - 2*l1*m1*s01*dth0*dth1 - l1*m1*s01*dth1**2 - l1*m2*s01*dth0**2 - 2*l1*m2*s01*dth0*dth1 - l1*m2*s01*dth1**2 - l2*m2*s012*dth0**2 - 2*l2*m2*s012*dth0*dth1 - 2*l2*m2*s012*dth0*dth2 - l2*m2*s012*dth1**2 - 2*l2*m2*s012*dth1*dth2 - l2*m2*s012*dth2**2],   # -fy0
                [c0*g*l0*m0 + c0*g*l0*m1 + c0*g*l0*m2 + c01*g*l1*m1 + c01*g*l1*m2 + c012*g*l2*m2 - 2*l0*l1*m1*s1*dth0*dth1 - l0*l1*m1*s1*dth1**2 - 2*l0*l1*m2*s1*dth0*dth1 - l0*l1*m2*s1*dth1**2 - 2*l0*l2*m2*s12*dth0*dth1 - 2*l0*l2*m2*s12*dth0*dth2 - l0*l2*m2*s12*dth1**2 - 2*l0*l2*m2*s12*dth1*dth2 - l0*l2*m2*s12*dth2**2 - 2*l1*l2*m2*s2*dth0*dth2 - 2*l1*l2*m2*s2*dth1*dth2 - l1*l2*m2*s2*dth2**2 - tau0],
                [                                                                                                                                                               1.0*c01*g*l1*m1 + 1.0*c01*g*l1*m2 + 1.0*c012*g*l2*m2 + 1.0*l0*l1*m1*s1*dth0**2 + 1.0*l0*l1*m2*s1*dth0**2 + 1.0*l0*l2*m2*s12*dth0**2 - 2.0*l1*l2*m2*s2*dth0*dth2 - 2.0*l1*l2*m2*s2*dth1*dth2 - 1.0*l1*l2*m2*s2*dth2**2 - 1.0*tau1],
                [                                                                                                                                                                                                                                                         1.0*c012*g*l2*m2 + 1.0*l0*l2*m2*s12*dth0**2 + 1.0*l1*l2*m2*s2*dth0**2 + 2.0*l1*l2*m2*s2*dth0*dth1 + 1.0*l1*l2*m2*s2*dth1**2 - 1.0*tau2]
            ])
    
    ds = np.zeros_like(s)
    ds[IDX_x0] = s[IDX_dx]
    ds[IDX_y0] = s[IDX_dy]
    ds[IDX_th0] = s[IDX_dth0]
    ds[IDX_th1] = s[IDX_dth1]
    ds[IDX_th2] = s[IDX_dth2]

    dd = np.zeros(5)

    #assert y0 >= 0

    #if y0 == 0: # foot is contact
    #    if np.linalg.matrix_rank(A[2:,2:]) != 3:
    #        print(A[2:,2:])
    #        assert False
    #    dd[2:] = np.linalg.solve(A[2:,2:], b[2:]).reshape(3)
    #    fxy = b[:2].reshape((2,)) - A[:2,2:] @ dd[2:]
    #    fy = fxy[1]
    #    #if fy > 0: # jump start
    #    #    if np.linalg.matrix_rank(A) != 5:
    #    #        print(A, np.linalg.matrix_rank(A) )
    #    #        assert False
    #    #    dd = np.linalg.solve(A,b)
    #else: # foot in the air
    #    dd = np.linalg.solve(A, b)

    ds[IDX_dx]   = dd[0]
    ds[IDX_dy]   = dd[1]
    ds[IDX_dth0] = dd[2]
    ds[IDX_dth1] = dd[3]
    ds[IDX_dth2] = dd[4]

    return ds


def obs(s):
    # sensor 6-IMU? estimated th0 is noisy...
    return np.concatenate([s[IDX_th0:IDX_th2+1], s[IDX_dth0:IDX_dth2+1]], axis=-1)

ob_low = np.array([ 0
                  , 0
                  , -np.pi
                  , -8
                  , -8
                  , -8
          ], dtype=np.float32)

ob_high = np.array([ np.pi
                   , np.pi
                   , np.pi
                   , 8
                   , 8
                   , 8
            ], dtype=np.float32)

def clip(s):
    new_s = s.copy()
    new_s[IDX_th0:IDX_th2+1] = np.clip(new_s[IDX_th0:IDX_th2+1], ob_low[0:3], ob_high[0:3])
    new_s[IDX_dth0:IDX_dth2+1] = np.clip(new_s[IDX_dth0:IDX_dth2+1], ob_low[3:6], ob_high[3:6])
    new_s[IDX_y0] = np.max(new_s[IDX_y0], 0)
    return new_s


def step(model, s, u):
    T = np.array([0, dt])
    u = np.repeat(np.array(u).reshape(Nu,1), 2, axis=1)
    t, s = ct.input_output_response(model, T, U=u, X0=s, params={})
    s = clip(s[:,-1])
    return s


def reset_state(np_random=None):
    s = np.zeros(10, dtype=np.float32)
    s[IDX_x0]   = 0.
    s[IDX_y0]   = 0.
    s[IDX_th0]  = np.pi/2
    s[IDX_th1]  = np.pi/2
    s[IDX_th2]  = -np.pi/3
    s[IDX_dx  ] = 0.
    s[IDX_dy  ] = 0.
    s[IDX_dth0] = 0.01
    s[IDX_dth1] = 0.
    s[IDX_dth2] = 0.

    if np_random is not None:
        s[IDX_th0] = np.pi/4  + np_random.uniform(low=-np.pi/10, high=np.pi/10)
        s[IDX_th1] = np.pi/2  + np_random.uniform(low=-np.pi/10, high=np.pi/10)
        s[IDX_th2] = -np.pi/3 + np_random.uniform(low=-np.pi/4, high=np.pi/4)
    return s


class RabbitEnv(gym.Env):

    metadata = {
        'render.modes' : ['human', 'rgb_array'],
        'video.frames_per_second' : 30
    }

    def __init__(self):
        self.model = ct.NonlinearIOSystem(rhs, outfcn=None
                        #, dt=dt
                        , inputs=('tau1', 'tau2')
                        , states=('x', 'y', 'th0', 'th1', 'th2', 'dx', 'dy', 'dth0', 'dth1', 'dth2')
                        , name='rabit')

        self.state = reset_state()
        self.viewer = None
        self.frame_no = 0

        max_action = np.array([MAX_TORQUE, MAX_TORQUE])
        self.action_space = spaces.Box(low=-max_action, high=max_action, dtype=np.float32)
        self.observation_space = spaces.Box(low=ob_low, high=ob_high, dtype=np.float32)

        self.seed()


    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def reward(self):
        #TODO
        return 0


    def step(self, u):
        self.state = step(self.model, self.state, u)
        self.frame_no = self.frame_no + 1
        return obs(self.state), 0, False, {}


    def reset(self):
        self.state = reset_state(self.np_random)
        self.frame_no = 0
        return obs(self.state)


    def render(self, mode='human'):

        if self.viewer is None:
            from gym.envs.classic_control import rendering
            self.viewer = rendering.Viewer(500,500)
            self.viewer.set_bounds(-2.2,2.2,-2.2,2.2)

            rod0 = rendering.make_capsule(l0, .2)
            rod0.set_color(.0, .0, .0)
            self.t0 = rendering.Transform()
            rod0.add_attr(self.t0)
            self.viewer.add_geom(rod0)

            rod1 = rendering.make_capsule(l1, .2)
            rod1.set_color(.0, .0, .0)
            self.t1 = rendering.Transform()
            rod1.add_attr(self.t1)
            self.viewer.add_geom(rod1)

            #rod2 = rendering.make_capsule(l2, .2)
            #rod2.set_color(.0, .0, .0)
            #self.t2 = rendering.Transform()
            #rod2.add_attr(self.t2)
            #self.viewer.add_geom(rod2)

            #foot = rendering.make_circle(.05)
            #foot.set_color(0,0,0)
            #self.viewer.add_geom(foot)

        print(self.state[IDX_th0])
        print(self.state[IDX_dth0])
        offset_t = np.array([0, RENDER_OFFSET_Y]) + self.state[IDX_x0:IDX_y0+1]
        offset_r = self.state[IDX_th0]
        self.t0.set_rotation(offset_r)
        self.t0.set_translation(offset_t[0], offset_t[1])

        offset_t = offset_t + np.array([np.cos(offset_r), np.sin(offset_r)])
        offset_r = offset_r + self.state[IDX_th1]
        self.t1.set_rotation(offset_r)
        self.t1.set_translation(offset_t[0], offset_t[1])

        #self.t2.set_rotation(self.state[0] + np.pi/2)
        #self.t2.set_translation(self.state[0] + np.pi/2)

        #if self.frame_no < 3:
        #    time.sleep(1)
        #time.sleep(0.1)

        return self.viewer.render(return_rgb_array = mode=='rgb_array')

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None

if __name__ == '__main__':
    env = RabbitEnv()
    while True:
        env.reset()
        print(env.state)
        for i in range(1000):
            env.step(np.array([0, 0]))
            env.render()












