
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

m0 = 0.1
m1 = 1
m2 = 0.5
m3 = 1


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
    fx0  = 0
    fy0  = 0
    tau0 = 0
    tau1 = u[0]
    tau2 = u[1]
    extf = np.array([fx0, fy0, tau0, tau1, tau2]).reshape(5,1)

    y0 = max(y0, 0)
    assert y0 >= 0

    A = np.array([ [m3+m2+m1+m0,0,(-l0*m3*sin(th0))-l0*m2*sin(th0)-l0*m1*sin(th0),(-l1*m3*sin(th1))-l1*m2*sin(th1),-l2*m3*sin(th2)]
                 , [0,m3+m2+m1+m0,l0*m3*cos(th0)+l0*m2*cos(th0)+l0*m1*cos(th0),l1*m3*cos(th1)+l1*m2*cos(th1),l2*m3*cos(th2)]
                 , [(-l0*m3*sin(th0))-l0*m2*sin(th0)-l0*m1*sin(th0),l0*m3*cos(th0)+l0*m2*cos(th0)+l0*m1*cos(th0),l0**2*m3+l0**2*m2+l0**2*m1,l0*l1*m3*sin(th0)*sin(th1)+l0*l1*m2*sin(th0)*sin(th1)+l0*l1*m3*cos(th0)*cos(th1)+l0*l1*m2*cos(th0)*cos(th1),l0*l2*m3*sin(th0)*sin(th2)+l0*l2*m3*cos(th0)*cos(th2)]
                 , [(-l1*m3*sin(th1))-l1*m2*sin(th1),l1*m3*cos(th1)+l1*m2*cos(th1),l0*l1*m3*sin(th0)*sin(th1)+l0*l1*m2*sin(th0)*sin(th1)+l0*l1*m3*cos(th0)*cos(th1)+l0*l1*m2*cos(th0)*cos(th1),l1**2*m3+l1**2*m2,l1*l2*m3*sin(th1)*sin(th2)+l1*l2*m3*cos(th1)*cos(th2)]
                 , [-l2*m3*sin(th2),l2*m3*cos(th2),l0*l2*m3*sin(th0)*sin(th2)+l0*l2*m3*cos(th0)*cos(th2),l1*l2*m3*sin(th1)*sin(th2)+l1*l2*m3*cos(th1)*cos(th2),l2**2*m3]
                 ])

    b = np.array([ [(-dth2**2*l2*m3*cos(th2))-dth1**2*l1*m3*cos(th1)-dth1**2*l1*m2*cos(th1)-dth0**2*l0*m3*cos(th0)-dth0**2*l0*m2*cos(th0)-dth0**2*l0*m1*cos(th0)]
                 , [(-dth2**2*l2*m3*sin(th2))-dth1**2*l1*m3*sin(th1)-dth1**2*l1*m2*sin(th1)-dth0**2*l0*m3*sin(th0)-dth0**2*l0*m2*sin(th0)-dth0**2*l0*m1*sin(th0)+g*m3+g*m2+g*m1+g*m0]
                 , [(-dth2**2*l0*l2*m3*sin(th2-th0))-dth1**2*l0*l1*m3*sin(th1-th0)-dth1**2*l0*l1*m2*sin(th1-th0)+g*l0*m3*cos(th0)+g*l0*m2*cos(th0)+g*l0*m1*cos(th0)]
                 , [(-dth2**2*l1*l2*m3*sin(th2-th1))+dth0**2*l0*l1*m3*sin(th1-th0)+dth0**2*l0*l1*m2*sin(th1-th0)+g*l1*m3*cos(th1)+g*l1*m2*cos(th1)]
                 , [dth1**2*l1*l2*m3*sin(th2-th1)+dth0**2*l0*l2*m3*sin(th2-th0)+g*l2*m3*cos(th2)]
                 ])

    ds = np.zeros_like(s)
    ds[IDX_x0] = s[IDX_dx]
    ds[IDX_y0] = s[IDX_dy]
    ds[IDX_th0] = s[IDX_dth0]
    ds[IDX_th1] = s[IDX_dth1]
    ds[IDX_th2] = s[IDX_dth2]

    assert np.linalg.matrix_rank(A) == 5, (s, A)

    dd = np.linalg.solve(A, extf-b).reshape(5)

    if y0 == 0 and dd[1] <= 0: # foot is contact
        #print("{:0.2f} foot is contact to the floor".format(t))
        dd = np.zeros(5)
        ds[IDX_x0] = 0
        ds[IDX_y0] = 0
        assert np.linalg.matrix_rank(A[2:,2:]) == 3
        ddtheta = np.linalg.solve(A[2:,2:], extf[2:]-b[2:]).reshape(3)
        dd[2:] = ddtheta
        #fxy = b[:2].reshape((2,)) - A[:2,2:] @ dd[2:]
        #fy = fxy[1]
    else:
        pass
        #print("{:0.2f} foot is in the air".format(t))


    ds[IDX_dx]   = dd[0]
    ds[IDX_dy]   = dd[1]
    ds[IDX_dth0] = dd[2]
    ds[IDX_dth1] = dd[3]
    ds[IDX_dth2] = dd[4]

    return ds

systemc = ct.NonlinearIOSystem(rhs, outfcn=None
        #, dt=dt
        , inputs=('tau1', 'tau2')
        , states=('x', 'y', 'th0', 'th1', 'th2', 'dx', 'dy', 'dth0', 'dth1', 'dth2')
        , name='rabit')


def rk4(f, t, s, u, params, dt):

    k1 = f(t,        s,             u, params)
    k2 = f(t + dt/2, s + dt/2 * k1, u, params)
    k3 = f(t + dt/2, s + dt/2 * k2, u, params)
    k4 = f(t + dt,   s + dt * k3,   u, params)
    print(k1, k2, k3, k4)

    return s + (k1 + 2*k2 + 2*k3 + k4)/6 * dt

def rhsd(t, s, u, params={}):
    if False:
        ds = rhs(t, s, u, params)
        s = s + dt * ds
    else:
        s = rk4(rhs, t, s, u, params, dt)
        print(s)
    s[IDX_y0] = max(s[IDX_y0], 0)
    return s

systemd = ct.NonlinearIOSystem(rhsd, outfcn=None
        , dt=dt
        , inputs=('tau1', 'tau2')
        , states=('x', 'y', 'th0', 'th1', 'th2', 'dx', 'dy', 'dth0', 'dth1', 'dth2')
        , name='rabit')

def node_pos(s):
    th0 = s[IDX_th0]
    th1 = s[IDX_th1]
    th2 = s[IDX_th2]

    p0 = s[IDX_x0:IDX_y0+1].copy()

    p1 = p0 + l0 * np.array([np.cos(th0), np.sin(th0)])

    p2 = p1 + l1 * np.array([np.cos(th1), np.sin(th1)])

    p3 = p2 + l2 * np.array([np.cos(th2), np.sin(th2)])

    return p0, p1, p2, p3

def node_vel(s):
    th0 = s[IDX_th0]
    th1 = s[IDX_th1]
    th2 = s[IDX_th2]
    dth0 = s[IDX_dth0]
    dth1 = s[IDX_dth1]
    dth2 = s[IDX_dth2]
    v0 = s[IDX_dx:IDX_dy+1]
    v1 = v0 + np.array([-sin(th0), cos(th0)]) * l0 * dth0
    v2 = v1 + np.array([-sin(th1), cos(th1)]) * l1 * dth1
    v3 = v2 + np.array([-sin(th2), cos(th2)]) * l2 * dth2

    return v0, v1, v2, v3

def show(t, s):
    p0, p1, p2, p3 = node_pos(s)
    v0, v1, v2, v3 = node_vel(s)
    energy = p0[1] * m0 * g + p1[1] * m1 * g + p2[1] * m2 * g + p3[1] * m3 * g + m0 * v0 @ v0 / 2 + m1 * v1 @ v1 / 2 + m2 * v2 @ v2 / 2 + m3 * v3 @ v3 / 2
    print("{:.4f} {:.4f}".format(t, energy))


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
    #new_s[IDX_th0:IDX_th2+1] = np.clip(new_s[IDX_th0:IDX_th2+1], ob_low[0:3], ob_high[0:3])
    #new_s[IDX_dth0:IDX_dth2+1] = np.clip(new_s[IDX_dth0:IDX_dth2+1], ob_low[3:6], ob_high[3:6])
    new_s[IDX_y0] = np.max(new_s[IDX_y0], 0)
    return new_s


def step(system, s, u, t=0):
    T = np.array([t, t+dt])
    u = np.repeat(np.array(u).reshape(Nu,1), 2, axis=1)
    t, s = ct.input_output_response(system, T, U=u, X0=s, params={})
    return s[:, -1]

def constant_steps(system, s, u, T):
    u = np.repeat(np.array(u).reshape(Nu,1), T.shape[0], axis=1)
    t, s = ct.input_output_response(system, T, U=u, X0=s, params={})
    return s


def reset_state(np_random=None):
    s = np.zeros(10, dtype=np.float32)
    s[IDX_x0]   = 0.
    s[IDX_y0]   = 0.
    s[IDX_th0]  = np.pi/4
    s[IDX_th1]  = np.pi*3/4
    s[IDX_th2]  = np.pi*5/12
    s[IDX_dx  ] = 0.
    s[IDX_dy  ] = 0.
    s[IDX_dth0] = 0.01
    s[IDX_dth1] = 0.01
    s[IDX_dth2] = 0.01

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

        horizon = rendering.Line((-500, 0), (500, 0))
        self.viewer.add_geom(horizon)


    def render(self, state):

        img_scale = 0.3
        p0, p1, p2, p3 = node_pos(state)

        th0 = state[IDX_th0]
        th1 = state[IDX_th1]
        th2 = state[IDX_th2]
        #self.viewer.add_onetime(self.img1)
        #self.viewer.add_onetime(self.img2)

        self.t0.set_rotation(th0)
        self.t0.set_translation(p0[0], p0[1])

        self.t1.set_rotation(th1)
        self.t1.set_translation(p1[0], p1[1])
        #self.it1.scale = (last_u[0]*img_scale, np.abs(last_u[0]*img_scale))
        #self.it1.set_translation(p1[0], p1[1])

        self.t2.set_rotation(th2)
        self.t2.set_translation(p2[0], p2[1])
        #self.it2.scale = (last_u[1]*img_scale, np.abs(last_u[1]*img_scale))
        #self.it2.set_translation(p2[0], p2[1])

        self.t3.set_translation(p3[0], p3[1])

        #if self.frame_no < 3:
        #    time.sleep(1)
        #time.sleep(0.1)

        return self.viewer.render(return_rgb_array = False)

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None


#class RabbitEnv(gym.Env):
#
#    metadata = {
#        'render.modes' : ['human', 'rgb_array'],
#        'video.frames_per_second' : 30
#    }
#
#    def __init__(self):
#        self.system = systemd
#
#        self.state = reset_state()
#        self.viewer = None
#
#        max_action = np.array([MAX_TORQUE, MAX_TORQUE])
#        self.action_space = spaces.Box(low=-max_action, high=max_action, dtype=np.float32)
#        self.observation_space = spaces.Box(low=ob_low, high=ob_high, dtype=np.float32)
#
#        self.seed()
#
#    def seed(self, seed=None):
#        self.np_random, seed = seeding.np_random(seed)
#        return [seed]
#
#    def reward(self):
#        #TODO
#        return 0
#
#    def step(self, u):
#        self.state = step(self.system, self.state, u)
#        return obs(self.state), 0, False, {}
#
#
#    def reset(self):
#        self.state = reset_state(self.np_random)
#        return obs(self.state)
#
#
#    def render(self, mode='human'):
#        if self.viewer is None:
#            self.viewer = RabbitViewer()
#        return self.viewer.render(self.state)
#
#    def close(self):
#        if self.viewer:
#            self.viewer.close()
#            self.viewer = None

if __name__ == '__main__':
    state = reset_state()
    T = np.arange(0, 20, dt)
    u = np.array([0, 0])

    #history = constant_steps(systemc, state, u, T)
    #org2 = RabbitViewer()
    t = 0
    #for i in range(history.shape[1]):
    #    org2.render(history[:,i])
    #    show(t, history[:,i])
    #    time.sleep(dt*5)
    #    t = t + dt

    org1 = RabbitViewer()
    while True:
        state = rhsd(t, state, u)
        state = clip(state)
        org1.render(state)
        #show(t, state)
        time.sleep(dt*5)
        t = t + dt



    ##history = constant_steps(systemd, state, u, T)
    #org1 = RabbitViewer()
    ##org2 = RabbitViewer()

    #t = 0
    #while True:
    #    state = rhsd(t, state, u)
    #    state = clip(state)
    #    org1.render(state)
    #    #org2.render(history[:,i])
    #    time.sleep(0.1)
    #    t = t + dt



