
import pygame
import pygame.locals as pl
import gym
from gym import spaces
from gym.utils import seeding
import numpy as np
from numpy import sin, cos
from os import path
import time
import control as ct
import sys

pygame.init()
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
RED = (128, 0, 0)
BLUE = (0, 128, 0)
GREEN = (0, 0, 128)
GRAY = (80, 80, 80)
SCREEN_SIZE=(1300, 500)
SCALE=60

font = pygame.font.SysFont('Calibri', 25, True, False)

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

    return (k1 + 2*k2 + 2*k3 + k4)/6

def rhsd(t, s, u, params={}):
    s = s + rk4(rhs, t, s, u, params, dt) * dt
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

def energy(s):
    p0, p1, p2, p3 = node_pos(s)
    v0, v1, v2, v3 = node_vel(s)
    return p0[1] * m0 * g + p1[1] * m1 * g + p2[1] * m2 * g + p3[1] * m3 * g + m0 * v0 @ v0 / 2 + m1 * v1 @ v1 / 2 + m2 * v2 @ v2 / 2 + m3 * v3 @ v3 / 2


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
    new_s[IDX_y0] = max(new_s[IDX_y0], 0)
    if new_s[IDX_y0] == 0:
        new_s[IDX_dy] = 0
        new_s[IDX_dx] = 0
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


def flip(p):
    ret = p.copy()
    ret[1] = -ret[1]
    return ret

def conv(p):
    ret = flip(SCALE * p) + np.array(SCREEN_SIZE)/2
    return ret

class RabbitViewer():
    def __init__(self):

        self.screen = pygame.display.set_mode(SCREEN_SIZE)
        pygame.display.set_caption("rabbot-4-mass-point")
        self.clock = pygame.time.Clock()


    def render(self, state, ds, t):

        p0, p1, p2, p3 = node_pos(state)

        p0o = conv(p0)
        p1o = conv(p1)
        p2o = conv(p2)
        p3o = conv(p3)

        th0 = state[IDX_th0]
        th1 = state[IDX_th1]
        th2 = state[IDX_th2]

        self.screen.fill(WHITE)

        pygame.draw.circle(self.screen, GRAY , p0o, 5)
        pygame.draw.circle(self.screen, BLUE , p1o, 10)
        pygame.draw.circle(self.screen, GREEN, p2o, 10)
        pygame.draw.circle(self.screen, RED  , p3o, 20)
        pygame.draw.line(self.screen, BLACK, p0o, p1o, width=3)
        pygame.draw.line(self.screen, BLACK, p1o, p2o, width=3)
        pygame.draw.line(self.screen, BLACK, p2o, p3o, width=3)

        pygame.draw.line(self.screen, BLACK, [0,SCREEN_SIZE[1]/2], [SCREEN_SIZE[0], SCREEN_SIZE[1]/2])
        text = font.render("t={:.02f} E={:.01f} y0={:.02f} ddy0={:.02f} ".format(t,energy(state), p0[1], ds[IDX_dy]), True, BLACK)
        self.screen.blit(text, [300, 50])
        pygame.display.flip()
        self.clock.tick(60)

    def close(self):
        pygame.quit()
        #if self.viewer:
        #    self.viewer.close()
        #    self.viewer = None


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
    start = False
    slowrate = 1
    pygame.event.clear()
    while True:
        ds = rk4(rhs, t, state, u, {}, dt)
        for event in pygame.event.get():
            if event.type == pl.QUIT:
                org1.close()
                sys.exit()
            if event.type == pl.KEYDOWN:
                keyname = pygame.key.name(event.key)
                if keyname == 'q':
                    print("??")
                    org1.close()
                    sys.exit()
                elif keyname == 's':
                    start = True
                    state = reset_state()
                    t = 0
                elif keyname == 'd':
                    slowrate = 20
                elif keyname == 'u':
                    slowrate = 1
            elif event.type == pl.MOUSEBUTTONDOWN:
                start = start ^ True

        org1.render(state, ds, t)
        time.sleep(slowrate * dt)
        if start:
            state = state + ds * dt
            #state[IDX_y0] = max(state[IDX_y0], 0)
            state = clip(state)
            #show(t, state)
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



