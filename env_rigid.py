
import gym
from gym import spaces
from gym.utils import seeding
import numpy as np
from numpy import sin, cos
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
RENDER_OFFSET_Y = 0

MAX_TORQUE=2.

l0 =  1.
l1 =  1.
l2 =  1.

g  = 9.8

m0 = 1.
m1 = 0.5
m2 = 1.

I0 = 2.
I1 = 1.
I2 = 2.


def rhs(t, s, u, params={}):
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
    fx0  = 0
    fy0  = 0
    tau0 = 0
    tau1 = u[0]
    tau2 = u[1]
    extf = np.array([fx0, fy0, tau0, tau1, tau2]).reshape(5,1)

    y0 = max(y0, 0)
    assert y0 >= 0

    A = np.array([ [m2+m1+m0,0, m2*((-l2*sin(th2+th1+th0))-l1*sin(th1+th0)-l0*sin(th0)) +(m1*((-l1*sin(th1+th0))-2*l0*sin(th0)))/2-(l0*m0*sin(th0))/2, m2*((-l2*sin(th2+th1+th0))-l1*sin(th1+th0))-(l1*m1*sin(th1+th0))/2, -l2*m2*sin(th2+th1+th0)]
                 , [0,m2+m1+m0, m2*(l2*cos(th2+th1+th0)+l1*cos(th1+th0)+l0*cos(th0)) +(m1*(l1*cos(th1+th0)+2*l0*cos(th0)))/2+(l0*m0*cos(th0))/2, m2*(l2*cos(th2+th1+th0)+l1*cos(th1+th0))+(l1*m1*cos(th1+th0))/2, l2*m2*cos(th2+th1+th0)]
                 , [m2*((-l2*sin(th2+th1+th0))-l1*sin(th1+th0)-l0*sin(th0)) +(m1*((-l1*sin(th1+th0))-2*l0*sin(th0)))/2-(l0*m0*sin(th0))/2, m2*(l2*cos(th2+th1+th0)+l1*cos(th1+th0)+l0*cos(th0)) +(m1*(l1*cos(th1+th0)+2*l0*cos(th0)))/2+(l0*m0*cos(th0))/2, (m2*(2*((-l2*sin(th2+th1+th0))-l1*sin(th1+th0)-l0*sin(th0))**2 +2*(l2*cos(th2+th1+th0)+l1*cos(th1+th0)+l0*cos(th0))**2)) /2 +(m1*(((-l1*sin(th1+th0))-2*l0*sin(th0))**2/2 +(l1*cos(th1+th0)+2*l0*cos(th0))**2/2)) /2+(m0*((l0**2*sin(th0)**2)/2+(l0**2*cos(th0)**2)/2))/2+I2+I1+I0, (m2*(2*((-l2*sin(th2+th1+th0))-l1*sin(th1+th0)) *((-l2*sin(th2+th1+th0))-l1*sin(th1+th0)-l0*sin(th0)) +2*(l2*cos(th2+th1+th0)+l1*cos(th1+th0)) *(l2*cos(th2+th1+th0)+l1*cos(th1+th0)+l0*cos(th0)))) /2 +(m1*((l1*cos(th1+th0)*(l1*cos(th1+th0)+2*l0*cos(th0)))/2 -(l1*sin(th1+th0)*((-l1*sin(th1+th0))-2*l0*sin(th0)))/2)) /2+I2+I1, (m2*(2*l2*cos(th2+th1+th0) *(l2*cos(th2+th1+th0)+l1*cos(th1+th0)+l0*cos(th0)) -2*l2*sin(th2+th1+th0) *((-l2*sin(th2+th1+th0))-l1*sin(th1+th0)-l0*sin(th0)))) /2 +I2]
                 , [m2*((-l2*sin(th2+th1+th0))-l1*sin(th1+th0))-(l1*m1*sin(th1+th0))/2, m2*(l2*cos(th2+th1+th0)+l1*cos(th1+th0))+(l1*m1*cos(th1+th0))/2, (m2*(2*((-l2*sin(th2+th1+th0))-l1*sin(th1+th0)) *((-l2*sin(th2+th1+th0))-l1*sin(th1+th0)-l0*sin(th0)) +2*(l2*cos(th2+th1+th0)+l1*cos(th1+th0)) *(l2*cos(th2+th1+th0)+l1*cos(th1+th0)+l0*cos(th0)))) /2 +(m1*((l1*cos(th1+th0)*(l1*cos(th1+th0)+2*l0*cos(th0)))/2 -(l1*sin(th1+th0)*((-l1*sin(th1+th0))-2*l0*sin(th0)))/2)) /2+I2+I1, (m2*(2*((-l2*sin(th2+th1+th0))-l1*sin(th1+th0))**2 +2*(l2*cos(th2+th1+th0)+l1*cos(th1+th0))**2)) /2 +(m1*((l1**2*sin(th1+th0)**2)/2+(l1**2*cos(th1+th0)**2)/2))/2+I2+I1, (m2*(2*l2*cos(th2+th1+th0)*(l2*cos(th2+th1+th0)+l1*cos(th1+th0)) -2*l2*sin(th2+th1+th0)*((-l2*sin(th2+th1+th0))-l1*sin(th1+th0)))) /2 +I2]
                 , [-l2*m2*sin(th2+th1+th0),l2*m2*cos(th2+th1+th0), (m2*(2*l2*cos(th2+th1+th0) *(l2*cos(th2+th1+th0)+l1*cos(th1+th0)+l0*cos(th0)) -2*l2*sin(th2+th1+th0) *((-l2*sin(th2+th1+th0))-l1*sin(th1+th0)-l0*sin(th0)))) /2 +I2, (m2*(2*l2*cos(th2+th1+th0)*(l2*cos(th2+th1+th0)+l1*cos(th1+th0)) -2*l2*sin(th2+th1+th0)*((-l2*sin(th2+th1+th0))-l1*sin(th1+th0)))) /2 +I2,(m2*(2*l2**2*sin(th2+th1+th0)**2+2*l2**2*cos(th2+th1+th0)**2))/2+I2]
                 ])

    b = np.array([ [(-dth2**2*l2*m2*cos(th2+th1+th0))-2*dth1*dth2*l2*m2*cos(th2+th1+th0) -2*dth0*dth2*l2*m2*cos(th2+th1+th0) -dth1**2*l2*m2*cos(th2+th1+th0) -2*dth0*dth1*l2*m2*cos(th2+th1+th0) -dth0**2*l2*m2*cos(th2+th1+th0) -dth1**2*l1*m2*cos(th1+th0) -2*dth0*dth1*l1*m2*cos(th1+th0) -dth0**2*l1*m2*cos(th1+th0) -(dth1**2*l1*m1*cos(th1+th0))/2 -dth0*dth1*l1*m1*cos(th1+th0) -(dth0**2*l1*m1*cos(th1+th0))/2 -dth0**2*l0*m2*cos(th0) -dth0**2*l0*m1*cos(th0) -(dth0**2*l0*m0*cos(th0))/2]
                 , [(-dth2**2*l2*m2*sin(th2+th1+th0))-2*dth1*dth2*l2*m2*sin(th2+th1+th0) -2*dth0*dth2*l2*m2*sin(th2+th1+th0) -dth1**2*l2*m2*sin(th2+th1+th0) -2*dth0*dth1*l2*m2*sin(th2+th1+th0) -dth0**2*l2*m2*sin(th2+th1+th0) -dth1**2*l1*m2*sin(th1+th0) -2*dth0*dth1*l1*m2*sin(th1+th0) -dth0**2*l1*m2*sin(th1+th0) -(dth1**2*l1*m1*sin(th1+th0))/2 -dth0*dth1*l1*m1*sin(th1+th0) -(dth0**2*l1*m1*sin(th1+th0))/2 -dth0**2*l0*m2*sin(th0) -dth0**2*l0*m1*sin(th0) -(dth0**2*l0*m0*sin(th0))/2+g*m2+g*m1 +g*m0]
                 , [(-dth2**2*l1*l2*m2*cos(th1+th0)*sin(th2+th1+th0)) -2*dth1*dth2*l1*l2*m2*cos(th1+th0)*sin(th2+th1+th0) -2*dth0*dth2*l1*l2*m2*cos(th1+th0)*sin(th2+th1+th0) -dth2**2*l0*l2*m2*cos(th0)*sin(th2+th1+th0) -2*dth1*dth2*l0*l2*m2*cos(th0)*sin(th2+th1+th0) -2*dth0*dth2*l0*l2*m2*cos(th0)*sin(th2+th1+th0) -dth1**2*l0*l2*m2*cos(th0)*sin(th2+th1+th0) -2*dth0*dth1*l0*l2*m2*cos(th0)*sin(th2+th1+th0) +dth2**2*l1*l2*m2*sin(th1+th0)*cos(th2+th1+th0) +2*dth1*dth2*l1*l2*m2*sin(th1+th0)*cos(th2+th1+th0) +2*dth0*dth2*l1*l2*m2*sin(th1+th0)*cos(th2+th1+th0) +dth2**2*l0*l2*m2*sin(th0)*cos(th2+th1+th0) +2*dth1*dth2*l0*l2*m2*sin(th0)*cos(th2+th1+th0) +2*dth0*dth2*l0*l2*m2*sin(th0)*cos(th2+th1+th0) +dth1**2*l0*l2*m2*sin(th0)*cos(th2+th1+th0) +2*dth0*dth1*l0*l2*m2*sin(th0)*cos(th2+th1+th0) +g*l2*m2*cos(th2+th1+th0)-dth1**2*l0*l1*m2*cos(th0)*sin(th1+th0) -2*dth0*dth1*l0*l1*m2*cos(th0)*sin(th1+th0) -(dth1**2*l0*l1*m1*cos(th0)*sin(th1+th0))/2 -dth0*dth1*l0*l1*m1*cos(th0)*sin(th1+th0) +dth1**2*l0*l1*m2*sin(th0)*cos(th1+th0) +2*dth0*dth1*l0*l1*m2*sin(th0)*cos(th1+th0) +(dth1**2*l0*l1*m1*sin(th0)*cos(th1+th0))/2 +dth0*dth1*l0*l1*m1*sin(th0)*cos(th1+th0)+g*l1*m2*cos(th1+th0) +(g*l1*m1*cos(th1+th0))/2+g*l0*m2*cos(th0)+g*l0*m1*cos(th0) +(g*l0*m0*cos(th0))/2]
                 , [(-dth2**2*l1*l2*m2*cos(th1+th0)*sin(th2+th1+th0)) -2*dth1*dth2*l1*l2*m2*cos(th1+th0)*sin(th2+th1+th0) -2*dth0*dth2*l1*l2*m2*cos(th1+th0)*sin(th2+th1+th0) +dth0**2*l0*l2*m2*cos(th0)*sin(th2+th1+th0) +dth2**2*l1*l2*m2*sin(th1+th0)*cos(th2+th1+th0) +2*dth1*dth2*l1*l2*m2*sin(th1+th0)*cos(th2+th1+th0) +2*dth0*dth2*l1*l2*m2*sin(th1+th0)*cos(th2+th1+th0) -dth0**2*l0*l2*m2*sin(th0)*cos(th2+th1+th0)+g*l2*m2*cos(th2+th1+th0) +dth0**2*l0*l1*m2*cos(th0)*sin(th1+th0) +(dth0**2*l0*l1*m1*cos(th0)*sin(th1+th0))/2 -dth0**2*l0*l1*m2*sin(th0)*cos(th1+th0) -(dth0**2*l0*l1*m1*sin(th0)*cos(th1+th0))/2+g*l1*m2*cos(th1+th0) +(g*l1*m1*cos(th1+th0))/2]
                 , [dth1**2*l1*l2*m2*cos(th1+th0)*sin(th2+th1+th0) +2*dth0*dth1*l1*l2*m2*cos(th1+th0)*sin(th2+th1+th0) +dth0**2*l1*l2*m2*cos(th1+th0)*sin(th2+th1+th0) +dth0**2*l0*l2*m2*cos(th0)*sin(th2+th1+th0) -dth1**2*l1*l2*m2*sin(th1+th0)*cos(th2+th1+th0) -2*dth0*dth1*l1*l2*m2*sin(th1+th0)*cos(th2+th1+th0) -dth0**2*l1*l2*m2*sin(th1+th0)*cos(th2+th1+th0) -dth0**2*l0*l2*m2*sin(th0)*cos(th2+th1+th0)+g*l2*m2*cos(th2+th1+th0)] 
                 ])

    ds = np.zeros_like(s)
    ds[IDX_x0] = s[IDX_dx]
    ds[IDX_y0] = s[IDX_dy]
    ds[IDX_th0] = s[IDX_dth0]
    ds[IDX_th1] = s[IDX_dth1]
    ds[IDX_th2] = s[IDX_dth2]

    assert np.linalg.matrix_rank(A) == 5, (s, A)

    dd = np.linalg.solve(A, extf-b).reshape(5)

    #if True:
    if y0 == 0 and dd[1] <= 0: # foot is contact
        dd = np.zeros(5)
        assert np.linalg.matrix_rank(A[2:,2:]) == 3
        ddtheta = np.linalg.solve(A[2:,2:], extf[2:]-b[2:]).reshape(3)
        print("ddtheta", ddtheta)
        dd[2:] = ddtheta
        fxy = b[:2].reshape((2,)) - A[:2,2:] @ dd[2:]
        fy = fxy[1]

    ds[IDX_dx]   = dd[0]
    ds[IDX_dy]   = dd[1]
    ds[IDX_dth0] = dd[2]
    ds[IDX_dth1] = dd[3]
    ds[IDX_dth2] = dd[4]

    return ds

def show(s, ds, dt):
    th0  = s[IDX_th0]
    th1  = s[IDX_th1]
    th2  = s[IDX_th2]
    dth0 = s[IDX_dth0]   * dt
    dth1 = s[IDX_dth1]   * dt
    dth2 = s[IDX_dth2]   * dt
    print("{:.2f} {:.2f} {:.2f} : {:.2f} {:.2f} {:.2f} : {:.4f} {:.4f} {:.4f} : ".format(
        th0, th1, th2, dth0, dth1, dth2, ds[2],ds[3],ds[4]))

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
    return clip(s[:,-1])
    #return s[:,-1]

def constant_steps(model, s, u, T):
    u = np.repeat(np.array(u).reshape(Nu,1), T.shape[0], axis=1)
    t, s = ct.input_output_response(model, T, U=u, X0=s, params={})
    return s


def reset_state(np_random=None):
    s = np.zeros(10, dtype=np.float32)
    s[IDX_x0]   = 0.
    s[IDX_y0]   = 0.
    s[IDX_th0]  = np.pi/4
    s[IDX_th1]  = np.pi/2
    s[IDX_th2]  = -np.pi/3
    s[IDX_dx  ] = 0.
    s[IDX_dy  ] = 0.
    s[IDX_dth0] = 0.01
    s[IDX_dth1] = 0.
    s[IDX_dth2] = 0.

    #if np_random is not None:
    #    s[IDX_th0] = np.pi/4  + np_random.uniform(low=-np.pi/10, high=np.pi/10)
    #    s[IDX_th1] = np.pi/2  + np_random.uniform(low=-np.pi/10, high=np.pi/10)
    #    s[IDX_th2] = -np.pi/3 + np_random.uniform(low=-np.pi/4, high=np.pi/4)
    return s

class RabbitViewer():
    def __init__(self):
        from gym.envs.classic_control import rendering
        self.viewer = rendering.Viewer(500,500)
        wscale = 4.2
        self.viewer.set_bounds(-wscale,wscale,-wscale,wscale)

        fname = path.join(path.dirname(__file__), "clockwise.png")

        rod0 = rendering.make_capsule(l0, .2)
        rod0.set_color(.0, .3, .3)
        self.t0 = rendering.Transform()
        rod0.add_attr(self.t0)
        self.viewer.add_geom(rod0)

        self.img0 = rendering.Image(fname, 1., 1.)
        self.it0 = rendering.Transform()
        self.img0.add_attr(self.it0)

        rod1 = rendering.make_capsule(l1, .1)
        rod1.set_color(.2, .5, .0)
        self.t1 = rendering.Transform()
        rod1.add_attr(self.t1)
        self.viewer.add_geom(rod1)

        self.img1 = rendering.Image(fname, 1., 1.)
        self.it1 = rendering.Transform()
        self.img1.add_attr(self.it1)

        rod2 = rendering.make_capsule(l2, .05)
        rod2.set_color(.4, .0, .4)
        self.t2 = rendering.Transform()
        rod2.add_attr(self.t2)
        self.viewer.add_geom(rod2)

        self.img2 = rendering.Image(fname, .5, .5)
        self.it2 = rendering.Transform()
        self.img2.add_attr(self.it2)

        head = rendering.make_circle(.2)
        head.set_color(0.3,0.1,0.1)
        self.t3 = rendering.Transform()
        head.add_attr(self.t3)
        self.viewer.add_geom(head)

    def render(self, state):

        img_scale = 0.3
        #self.viewer.add_onetime(self.img0)
        #self.viewer.add_onetime(self.img1)
        #self.viewer.add_onetime(self.img2)

        offset_t = np.array([0, RENDER_OFFSET_Y]) + state[IDX_x0:IDX_y0+1]
        offset_r = state[IDX_th0]
        self.t0.set_rotation(offset_r)
        self.t0.set_translation(offset_t[0], offset_t[1])
        self.it0.scale = (-state[IDX_dth0]*img_scale, np.abs(state[IDX_dth0]*img_scale))
        self.it0.set_translation(offset_t[0], offset_t[1])

        offset_t = offset_t + l0 * np.array([np.cos(offset_r), np.sin(offset_r)])
        offset_r = offset_r + state[IDX_th1]
        self.t1.set_rotation(offset_r)
        self.t1.set_translation(offset_t[0], offset_t[1])
        self.it1.scale = (-state[IDX_dth1]*img_scale, np.abs(state[IDX_dth1]*img_scale))
        self.it1.set_translation(offset_t[0], offset_t[1])

        offset_t = offset_t + l1 * np.array([np.cos(offset_r), np.sin(offset_r)])
        offset_r = offset_r + state[IDX_th2]
        self.t2.set_rotation(offset_r)
        self.t2.set_translation(offset_t[0], offset_t[1])
        self.it2.scale = (-state[IDX_dth2]*img_scale, np.abs(state[IDX_dth2]*img_scale))
        self.it2.set_translation(offset_t[0], offset_t[1])

        offset_t = offset_t + l2 * np.array([np.cos(offset_r), np.sin(offset_r)])
        self.t3.set_translation(offset_t[0], offset_t[1])

        #if self.frame_no < 3:
        #    time.sleep(1)
        #time.sleep(0.1)

        return self.viewer.render(return_rgb_array = False)

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None

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
            self.viewer = RabbitViewer()

        return self.viewer.render(self.state)

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None

if __name__ == '__main__':
    env = RabbitEnv()
    env.reset()
    dt = 0.03
    T = np.arange(0, 20, dt)
    u = np.array([0, 0])

    history = constant_steps(env.model, env.state, u, T)
    org1 = RabbitViewer()
    #org2 = RabbitViewer()

    for i in range(T.shape[0]):
        s = step(env.model, env.state, u)
        print(s - history[:, i])
        org1.render(s)
        #org2.render( history[:,i])
        env.state = s
        #ds = rhs(0, s, u)
        #show(s, ds, dt)
        time.sleep(0.1)

    #for i in range(history.shape[1]):
    #    #if i == 1:
    #    #    input('')
    #    s = history[:,i]
    #    env.state = s
    #    ds = rhs(0, s, u)
    #    show(s, ds, dt)
    #    env.render()
    #    #time.sleep(0.03)
    #    time.sleep(0.1)


    #while True:
        #step(np.array([0, 0]))

        #for i in range(1000):
        #    env.step()












