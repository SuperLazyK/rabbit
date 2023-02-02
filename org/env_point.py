
import pygame
import pygame.locals as pl
import gym
from gym import spaces
from gym.utils import seeding
import numpy as np
from numpy import sin, cos
from os import path
import os
import time
import control as ct
import sys


#----------------------------
# Dynamics
#----------------------------
delta = 0.01

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

MAX_TORQUE1=40.0
MAX_TORQUE2=20.0
MAX_ANGV=10

l0 =  1.
l1 =  1.
l2 =  1.

g  = 9.8
#g  = 1

m0 = 0.1
m1 = 0.5
m2 = 0.5
m3 = 0.3

def calcAb(s, u):
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
    return A, b, extf


def rhs_free(t, s, u, params={}):

    A, b, extf = calcAb(s, u)

    ds = np.zeros_like(s)
    ds[IDX_x0] = s[IDX_dx]
    ds[IDX_y0] = s[IDX_dy]
    ds[IDX_th0] = s[IDX_dth0]
    ds[IDX_th1] = s[IDX_dth1]
    ds[IDX_th2] = s[IDX_dth2]

    assert np.linalg.matrix_rank(A) == 5, (s, A)

    dd = np.linalg.solve(A, extf-b).reshape(5)

    ds[IDX_dx]   = dd[0]
    ds[IDX_dy]   = dd[1]
    ds[IDX_dth0] = dd[2]
    ds[IDX_dth1] = dd[3]
    ds[IDX_dth2] = dd[4]

    return ds

def rhs_p0_fixed_constraint(t, s, u, params={}):

    A, b, extf = calcAb(s, u)

    ds = np.zeros_like(s)
    ds[IDX_x0] = 0
    ds[IDX_y0] = 0
    ds[IDX_th0] = s[IDX_dth0]
    ds[IDX_th1] = s[IDX_dth1]
    ds[IDX_th2] = s[IDX_dth2]

    assert np.linalg.matrix_rank(A[2:,2:]) == 3
    ddtheta = np.linalg.solve(A[2:,2:], extf[2:]-b[2:]).reshape(3)

    ds[IDX_dx]   = 0
    ds[IDX_dy]   = 0
    ds[IDX_dth0] = ddtheta[0]
    ds[IDX_dth1] = ddtheta[1]
    ds[IDX_dth2] = ddtheta[2]

    return ds


def collision_impulse(s, u): # only change velocity
    ddxdt = - s[IDX_dx]
    ddydt = - s[IDX_dy]
    A, b, extf = calcAb(s, u)

    offset = A[2:,:2] @ np.array([[ddxdt],[ddydt]])

    assert np.linalg.matrix_rank(A[2:,2:]) == 3
    ddthetadt = np.linalg.solve(A[2:,2:], -offset).reshape(3)

    new_s = s.copy()
    new_s[IDX_dx]   = 0
    new_s[IDX_dy]   = 0
    new_s[IDX_dth0] = s[IDX_dth0] + ddthetadt[0]
    new_s[IDX_dth1] = s[IDX_dth1] + ddthetadt[1]
    new_s[IDX_dth2] = s[IDX_dth2] + ddthetadt[2]

    return new_s



def rk4(f, t, s, u, params, dt):

    k1 = f(t,        s,             u, params)
    k2 = f(t + dt/2, s + dt/2 * k1, u, params)
    k3 = f(t + dt/2, s + dt/2 * k2, u, params)
    k4 = f(t + dt,   s + dt * k3,   u, params)

    return (k1 + 2*k2 + 2*k3 + k4)/6


def collision_time(s, ds, dt):
    s1 = s + ds * dt
    y0 = s[IDX_y0]
    y1 = s1[IDX_y0]
    if y0 > 0 and y1 <= 0:
        y0 = s[IDX_y0]
        y1 = s1[IDX_y0]
        return y0 * dt / (y0 - y1)
    else:
        return None

# rough approximation of torque
def torque_trade_off(tau, dtheta, max_t, max_v):
    a = (max_v - min(max_v, abs(dtheta))) * max_t / max_v
    if abs(tau) > a:
        return a if tau >= 0 else -a
    else:
        return tau

def limit_torque(s, u):
    ret = u.copy()
    #ret[0] = torque_trade_off(u[0], s[IDX_dth1], MAX_TORQUE1, MAX_ANGV)
    #ret[1] = torque_trade_off(u[1], s[IDX_dth2], MAX_TORQUE2, MAX_ANGV)
    return ret

def limit_th(s):
    new_s = s.copy()
    new_s[IDX_th0] = ((s[IDX_th0] + np.pi)% (2*np.pi)) - np.pi
    new_s[IDX_th1] = ((s[IDX_th1] + np.pi)% (2*np.pi)) - np.pi
    new_s[IDX_th2] = ((s[IDX_th2] + np.pi)% (2*np.pi)) - np.pi
    return new_s
    #return s

def step(t, s, u, dt):
    u = limit_torque(s, u)
    ds = rk4(rhs_free, t, s, u, {}, dt)
    if s[IDX_y0] == 0:
        if ds[IDX_y0] > 0:
            return "jump", t + dt, limit_th(s + ds * dt)
        else:
            ds = rk4(rhs_p0_fixed_constraint, t, s, u, {}, dt)
            return "constraint", t + dt, limit_th(s + ds * dt)
    elif s[IDX_y0] > 0:
        colt = collision_time(s, ds, dt)
        if colt is not None:
            ds = rk4(rhs_free, t, s, u, {}, colt)
            s = s + ds * colt
            s[IDX_y0] = 0
            s = collision_impulse(s, u)
            return  "collision", t + colt, limit_th(s)
        else:
            return "free", t + dt, limit_th(s + ds * dt)
    else:
        assert False

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

def energyU(s):
    p0, p1, p2, p3 = node_pos(s)
    return p0[1] * m0 * g + p1[1] * m1 * g + p2[1] * m2 * g + p3[1] * m3 * g

def energyT(s):
    v0, v1, v2, v3 = node_vel(s)
    return m0 * v0 @ v0 / 2 + m1 * v1 @ v1 / 2 + m2 * v2 @ v2 / 2 + m3 * v3 @ v3 / 2

def energy(s):
    return energyU(s) + energyT(s)

def print_energy(s):
    p0, p1, p2, p3 = node_pos(s)
    v0, v1, v2, v3 = node_vel(s)
    u0 = p0[1] * m0 * g
    u1 = p1[1] * m1 * g
    u2 = p2[1] * m2 * g
    u3 = p3[1] * m3 * g
    t0 = m0 * v0 @ v0 / 2
    t1 = m1 * v1 @ v1 / 2
    t2 = m2 * v2 @ v2 / 2
    t3 = m3 * v3 @ v3 / 2
    print("")
    print(f"u0:  {u0:.2f} u1:  {u1:.2f} u2:  {u2:.2f} u3:  {u3:.2f}")
    print(f"t0:  {t0:.2f} t1:  {t1:.2f} t2:  {t2:.2f} t3:  {t3:.2f}")
    print("")
    print(f"v0:  {v0} ")
    print(f"v1:  {v1} ")
    print(f"v2:  {v2} ")
    print(f"v3:  {v3} ")


def print_state(s):
    print("")
    print(f"x0:  {s[IDX_x0 ]:.2f} ")
    print(f"y0:  {s[IDX_y0 ]:.2f} ")
    print(f"th0: {s[IDX_th0]:.2f} ")
    print(f"th1: {s[IDX_th1]:.2f} ")
    print(f"th2: {s[IDX_th2]:.2f} ")
    print("")
    print(f"dx:  {s[IDX_dx  ]:.2f}")
    print(f"dy:  {s[IDX_dy  ]:.2f}")
    print(f"dth0:{s[IDX_dth0]:.2f}")
    print(f"dth1:{s[IDX_dth1]:.2f}")
    print(f"dth2:{s[IDX_dth2]:.2f}")


def print_info(mode, t, s, u,r):
    print(f"--{t:.2f}--{mode:}-- reward: {r:}")
    print_state(s)
    print_energy(s)
    print("")



#----------------------------
# Rendering
#----------------------------

pygame.init()

BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
YELLOW = (128, 128, 0)
RED = (128, 0, 0)
BLUE = (0, 128, 0)
GREEN = (0, 0, 128)
GRAY = (80, 80, 80)

VRED = (128, 30, 30)
VBLUE = (30, 128, 30)
VGREEN = (0, 30, 128)
VGRAY = (90, 90, 90)

SCREEN_SIZE=(1300, 500)
SCALE=50
RSCALE=1/SCALE

font = pygame.font.SysFont('Calibri', 25, True, False)


def flip(p):
    ret = p.copy()
    ret[1] = -ret[1]
    return ret

def conv_pos(p):
    ret = flip(SCALE * p) + np.array(SCREEN_SIZE)/2
    return ret

class RabbitViewer():
    def __init__(self):
        self.screen = pygame.display.set_mode(SCREEN_SIZE)
        pygame.display.set_caption("rabbot-4-mass-point")
        self.clock = pygame.time.Clock()
        self.rotate = pygame.image.load("clockwise.png")

        self.alter = pygame.image.load("clockwise.png")
        image_pixel_array = pygame.PixelArray(self.alter)
        image_pixel_array.replace(BLACK, RED)



    def render_rotation(self, u, pos, max_torq, max_scale=SCALE, default_img=True):
        a = abs(u)
        scale = (a*max_scale/max_torq, a*max_scale/max_torq)
        imgu = pygame.transform.scale(self.rotate if default_img else self.alter, scale)
        if u > 0:
            imgu = pygame.transform.flip(imgu, True, False)

        self.screen.blit(imgu, pos - np.array(scale)/2)


    def render(self, state, t, u, r):

        p0, p1, p2, p3 = node_pos(state)
        v0, v1, v2, v3 = node_vel(state)

        p0o = conv_pos(p0)
        p1o = conv_pos(p1)
        p2o = conv_pos(p2)
        p3o = conv_pos(p3)

        v0o = conv_pos(p0 + 20 * delta * v0)
        v1o = conv_pos(p1 + 20 * delta * v1)
        v2o = conv_pos(p2 + 20 * delta * v2)
        v3o = conv_pos(p3 + 20 * delta * v3)

        th0 = state[IDX_th0]
        th1 = state[IDX_th1]
        th2 = state[IDX_th2]

        dth0 = state[IDX_dth0]
        dth1 = state[IDX_dth1]
        dth2 = state[IDX_dth2]


        self.screen.fill(WHITE)

        pygame.draw.circle(self.screen, GRAY , p0o, 150 * RSCALE)
        pygame.draw.circle(self.screen, BLUE , p1o, 300 * RSCALE)
        pygame.draw.circle(self.screen, GREEN, p2o, 300 * RSCALE)
        pygame.draw.circle(self.screen, YELLOW  , p3o, 450 * RSCALE)
        pygame.draw.line(self.screen, BLACK, p0o, p1o, width=int(100 * RSCALE))
        pygame.draw.line(self.screen, BLACK, p1o, p2o, width=int(100 * RSCALE))
        pygame.draw.line(self.screen, BLACK, p2o, p3o, width=int(100 * RSCALE))

        #pygame.draw.line(self.screen, VGRAY , p0o, v0o, width=int(100 * RSCALE))
        #pygame.draw.line(self.screen, VBLUE , p1o, v1o, width=int(100 * RSCALE))
        #pygame.draw.line(self.screen, VGREEN, p2o, v2o, width=int(100 * RSCALE))
        #pygame.draw.line(self.screen, VRED  , p3o, v3o, width=int(100 * RSCALE))

        self.render_rotation(u[0], p1o + np.array([100, 0]), MAX_TORQUE1, default_img=False)
        self.render_rotation(u[1], p2o + np.array([100, 0]), MAX_TORQUE2, default_img=False)

        #self.render_rotation(dth0, p0o, MAX_ANGV)
        self.render_rotation(dth1-dth0, p1o, MAX_ANGV)
        self.render_rotation(dth2-dth1, p2o, MAX_ANGV)

        pygame.draw.line(self.screen, BLACK, [0,SCREEN_SIZE[1]/2], [SCREEN_SIZE[0], SCREEN_SIZE[1]/2])
        text = font.render("t={:.02f} E={:.01f} r={:.02f} ".format(t,energy(state), r), True, BLACK)
        self.screen.blit(text, [300, 50])
        pygame.display.flip()
        self.clock.tick(60)

    def close(self):
        pygame.quit()



#----------------------------
# Env (I/O + Reward)
#----------------------------

def pow2(v):
    return v @ v

def obs(s):
    # sensor 6-IMU? estimated th0 is noisy...
    return np.concatenate([s[IDX_th0:IDX_th2+1], s[IDX_dth0:IDX_dth2+1]], axis=-1)


def reset_state(np_random=None):
    s = np.zeros(10, dtype=np.float32)
    #s[IDX_x0]   = -10.
    s[IDX_x0]   = 0.
    s[IDX_y0]   = 0.
    s[IDX_th0]  = np.pi/4
    s[IDX_th1]  = np.pi*3/4
    s[IDX_th2]  = np.pi*5/12
    s[IDX_dx  ] = 0.
    s[IDX_dy  ] = 0.
    s[IDX_dth0] = 0
    s[IDX_dth1] = 0
    s[IDX_dth2] = 0

    #if np_random is not None:
    #    s[IDX_th0] = s[IDX_th0] + np_random.uniform(low=-np.pi/10, high=np.pi/10)
    #    s[IDX_th1] = s[IDX_th1] + np_random.uniform(low=-np.pi/10, high=np.pi/10)
    #    s[IDX_th2] = s[IDX_th2] + np_random.uniform(low=-np.pi/4, high=np.pi/4)

    return s

def game_over(s):
    p0, p1, p2, p3 = node_pos(s)
    if p0[1] < 0:
        #print(f"GAME OVER p0={p0:}")
        return True
    if p1[1] < 0:
        #print(f"GAME OVER p1={p1:}")
        return True
    if p2[1] < 0:
        #print(f"GAME OVER p2={p2:}")
        return True
    if p3[1] < 0:
        #print(f"GAME OVER p3={p3:}")
        return True
    return False


class RabbitEnv(gym.Env):

    metadata = {
        'render.modes' : ['human', 'rgb_array'],
        'video.frames_per_second' : 30
    }

    def __init__(self):
        self.reset(False)
        self.viewer = None

        max_action = np.array([MAX_TORQUE1, MAX_TORQUE2])
        max_obs    = np.array([ np.pi/2 , np.pi/2 , np.pi/2 , MAX_ANGV , MAX_ANGV , MAX_ANGV ], dtype=np.float32)

        self.action_space = spaces.Discrete(4)
        #self.action_space = spaces.Box(low=-max_action, high=max_action, dtype=np.float32)
        self.observation_space = spaces.Box(low=-max_obs, high=max_obs, dtype=np.float32)

        self.seed()

        self.step_render= int(os.environ.get('RENDER', "0"))

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, act):
        #u = u * np.array([MAX_TORQUE1, MAX_TORQUE2])
        if act == 0:
            u = np.array([-MAX_TORQUE1, -MAX_TORQUE2])
        elif act == 1:
            u = np.array([-MAX_TORQUE1, MAX_TORQUE2])
        elif act == 2:
            u = np.array([MAX_TORQUE1, -MAX_TORQUE2])
        else:
            u = np.array([MAX_TORQUE1, MAX_TORQUE2])
        _, t, s, _, _ = self.history[-1]
        mode, t, s = step(t, s, u, delta)
        th = s[IDX_th0:IDX_th2+1] - np.pi/2
        dth = s[IDX_dth0:IDX_dth2+1]
        p0, p1, p2, p3 = node_pos(s)
        c_x   = (p0[0]/20) ** 2
        #print("reward", r_x, r_att, r_vel, r_u, r_y)
        c_att = pow2(th/np.pi)
        c_vel = pow2(dth/MAX_ANGV)
        c_u   = pow2(u/np.array([MAX_TORQUE1, MAX_TORQUE2]))
        c_U = energyU(s)
        c_T = energyT(s)
        #print("reward", c_x, c_att, c_vel, c_u,)
        #r = 5 - 2*c_x - 1*c_u - 1*c_att - 1*c_vel
        #r = 5 - 2*c_x  - 1*c_att - 1*c_vel
        r =11 - 10*c_att - 1*c_vel - 0.1 * c_u
        #r = 30 + 4 * c_U - c_T
        done = game_over(s)

        if self.step_render != 0:
            self.render(-1)

        self.history.append((mode, t, s, u, r))

        return obs(s), r, done, {}

    def reset(self, random=True):
        print("reset")
        s = reset_state(self.np_random if random else None)
        self.history = [("start", 0, s, np.array([0,0]), 0)]
        return obs(s)

    def render(self, mode='human', frame=-1):
        print("render")
        if self.viewer is None:
            self.viewer = RabbitViewer()
        mode, t, s, u, r = self.history[frame]
        return self.viewer.render(s, t, u, r)

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None


