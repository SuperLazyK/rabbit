
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


#----------------------------
# Dynamics
#----------------------------

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
delta = 0.01

MAX_TORQUE=10.0
MAX_ANGV=30

l0 =  1.
l1 =  1.
l2 =  1.

g  = 9.8

m0 = 0.1
m1 = 1
m2 = 0.5
m3 = 1

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
def torque_trade_off(tau, dtheta):
    a = (MAX_ANGV - min(MAX_ANGV, abs(dtheta))) * MAX_TORQUE / MAX_ANGV
    if abs(tau) > a:
        return a if tau >= 0 else -a
    else:
        return tau

def limit_torque(s, u):
    ret = u.copy()
    ret[0] = torque_trade_off(u[0], s[IDX_dth1])
    ret[1] = torque_trade_off(u[1], s[IDX_dth2])
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
            return  "collision", limit_th(t + colt, s)
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


def energy(s):
    p0, p1, p2, p3 = node_pos(s)
    v0, v1, v2, v3 = node_vel(s)
    return p0[1] * m0 * g + p1[1] * m1 * g + p2[1] * m2 * g + p3[1] * m3 * g + m0 * v0 @ v0 / 2 + m1 * v1 @ v1 / 2 + m2 * v2 @ v2 / 2 + m3 * v3 @ v3 / 2

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


def print_info(mode, t, s, u):
    print(f"--{t:.2f}--{mode:}")
    print_state(s)
    print_energy(s)
    print("")



#----------------------------
# Rendering
#----------------------------



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


    def render(self, state, t):

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
        text = font.render("t={:.02f} E={:.01f} y0={:.02f} ".format(t,energy(state), p0[1]), True, BLACK)
        self.screen.blit(text, [300, 50])
        pygame.display.flip()
        self.clock.tick(60)

    def close(self):
        pygame.quit()



#----------------------------
# Env (I/O + Reward)
#----------------------------


def obs(s):
    # sensor 6-IMU? estimated th0 is noisy...
    return np.concatenate([s[IDX_th0:IDX_th2+1], s[IDX_dth0:IDX_dth2+1]], axis=-1)


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

    if np_random is not None:
        s[IDX_th0] = s[IDX_th0] + np_random.uniform(low=-np.pi/10, high=np.pi/10)
        s[IDX_th1] = s[IDX_th1] + np_random.uniform(low=-np.pi/10, high=np.pi/10)
        s[IDX_th2] = s[IDX_th2] + np_random.uniform(low=-np.pi/4, high=np.pi/4)

    return s

def game_over(s):
    p0, p1, p2, p3 = node_pos(s)
    if p0[1] < 0:
        print(f"GAME OVER p0={p0:}")
        return True
    if p1[1] < 0:
        print(f"GAME OVER p1={p1:}")
        return True
    if p2[1] < 0:
        print(f"GAME OVER p2={p2:}")
        return True
    if p3[1] < 0:
        print(f"GAME OVER p3={p3:}")
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

        max_action = np.array([MAX_TORQUE, MAX_TORQUE])
        max_obs    = np.array([ np.pi/2 , np.pi/2 , np.pi/2 , MAX_ANGV , MAX_ANGV , MAX_ANGV ], dtype=np.float32)

        self.action_space = spaces.Box(low=-max_action, high=max_action, dtype=np.float32)
        self.observation_space = spaces.Box(low=-max_obs, high=max_obs, dtype=np.float32)

        self.seed()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, u):
        t, s = self.history[-1]
        mode, t, s = step(t, s, u, delta)
        print_info(mode, t, s, u)
        self.history.append((t, s))
        return obs(s), self.reward(s,u), game_over(s), {}

    def reset(self, random=True):
        s = reset_state(self.np_random if random else None)
        self.history = [(0, s)]

    def render(self, mode='human', frame=-1):
        if self.viewer is None:
            self.viewer = RabbitViewer()
        print("render",frame)
        t, s = self.history[frame]
        return self.viewer.render(s, t)

    def reward(self, s, u):
        costs = 0 # angle_normalize(th) ** 2 + 0.1 * thdot**2 + 0.001 * (u**2)
        return -costs

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None


#----------------------------
# main
#----------------------------

def main():
    env = RabbitEnv()
    u = np.array([0, 0])
    wait_rate = 0
    frame = 0
    start = False
    pygame.event.clear()
    while True:
        n = len(env.history)
        for event in pygame.event.get():
            if event.type == pl.QUIT:
                env.close()
                sys.exit()
            elif event.type == pl.MOUSEBUTTONDOWN:
                start = start ^ True
            elif event.type == pl.KEYDOWN:
                keyname = pygame.key.name(event.key)
                mods = pygame.key.get_mods()
                #if keyname == 'q':
                #    env.close()
                #    sys.exit()
                if keyname == 'r':
                    frame = 0
                    s = env.reset(random=True)
                    n = len(env.history)
                elif keyname == 's':
                    start = start ^ True
                elif keyname == 'd':
                    wait_rate = min(20, wait_rate + 1)
                elif keyname == 'u':
                    wait_rate = max(0, wait_rate - 1)
                elif keyname == 'h':
                    u = -np.array([MAX_TORQUE, 0])
                elif keyname == 'l':
                    u = np.array([MAX_TORQUE, 0])
                elif keyname == 'j':
                    u = -np.array([0, MAX_TORQUE])
                elif keyname == 'k':
                    u = np.array([0, MAX_TORQUE])
                elif keyname == ';':
                    u = np.array([0, 0])
                elif keyname == 'n':
                    start = False
                    if mods & pl.KMOD_LSHIFT:
                        frame = min(frame + 10, n-1)
                    else:
                        frame = min(frame + 1, n-1)

                elif keyname == 'p':
                    start = False
                    if mods & pl.KMOD_LSHIFT:
                        frame = max(frame - 10, 0)
                    else:
                        frame = max(frame - 1, 0)

        env.render(frame= -1 if frame == n-1 else frame)
        if start:
            if frame == n-1:
                _, _, done, _ = env.step(u)
                if done:
                    start = False
            frame = frame + 1
        time.sleep(wait_rate * delta)

if __name__ == '__main__':
    main()
