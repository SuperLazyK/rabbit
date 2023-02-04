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
import model_spring_heeled_jack as mp

MAX_TORQUE1=40.0
MAX_TORQUE2=20.0
MAX_ANGV=10


#----------------------------
# Rendering
#----------------------------

delta = 0.005

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



    def render_rotation(self, u, pos, max_rot, max_scale=SCALE, default_img=True):
        a = min(abs(u)/max_rot,1)
        scale = (max_scale*a, max_scale*a)
        imgu = pygame.transform.scale(self.rotate if default_img else self.alter, scale)
        if u > 0:
            imgu = pygame.transform.flip(imgu, True, False)

        self.screen.blit(imgu, pos - np.array(scale)/2)


    def render(self, state, t, u, r):

        pr, p0, p1, p2, p3 = mp.node_pos(state)
        energy = mp.energy(state)

        pro = conv_pos(pr)
        p0o = conv_pos(p0)
        p1o = conv_pos(p1)
        p2o = conv_pos(p2)
        p3o = conv_pos(p3)

        self.screen.fill(WHITE)

        pygame.draw.circle(self.screen, GRAY , p0o, 150 * RSCALE)
        pygame.draw.circle(self.screen, BLUE , p1o, 300 * RSCALE)
        pygame.draw.circle(self.screen, GREEN, p2o, 300 * RSCALE)
        pygame.draw.circle(self.screen, YELLOW  , p3o, 450 * RSCALE)
        pygame.draw.line(self.screen, BLACK, p0o, pro, width=int(100 * RSCALE))
        pygame.draw.line(self.screen, BLACK, p0o, p1o, width=int(100 * RSCALE))
        pygame.draw.line(self.screen, BLACK, p1o, p2o, width=int(100 * RSCALE))
        pygame.draw.line(self.screen, BLACK, p2o, p3o, width=int(100 * RSCALE))


        #self.render_rotation(u[0], p1o, MAX_TORQUE1, default_img=False)
        #self.render_rotation(u[1], p2o, MAX_TORQUE2, default_img=False)

        #self.render_rotation(dth0, p0o, MAX_ANGV)
        #self.render_rotation(dth1, p1o, MAX_ANGV)
        #self.render_rotation(dth2, p2o, MAX_ANGV)
        pygame.draw.line(self.screen, BLACK, [0,SCREEN_SIZE[1]/2], [SCREEN_SIZE[0], SCREEN_SIZE[1]/2])
        text = font.render("t={:.02f} E={:.02f} r={:.02f} ".format(t, energy, r), True, BLACK)
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
    return s


def game_over(s):
    return False
    pr, p0, p1, p2, p3 = node_pos(s)
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

#----------------------------
# Cost
#----------------------------
def pow2(v):
    return v @ v

def stage_cost(s, u):
    
    #th = np.array(mp.node_angle(s)) - np.pi/2
    #dth = np.array(mp.node_omega(s))
    #pr, p0, p1, p2, p3 = mp.node_pos(s)
    #c_x   = (pr[0]/20) ** 2
    ##print("reward", r_x, r_att, r_vel, r_u, r_y)
    #c_att = pow2(th/np.pi)
    #c_vel = pow2(dth/MAX_ANGV)
    #c_u   = pow2(u)
    ##print("reward", c_x, c_att, c_vel, c_u,)
    ##r = 5 - 2*c_x - 1*c_u - 1*c_att - 1*c_vel
    ##r = 5 - 2*c_x  - 1*c_att - 1*c_vel
    #return - 10*c_att - 1*c_vel - 0.1 * c_u
    return 0

class RabbitEnv(gym.Env):

    metadata = {
        'render.modes' : ['human', 'rgb_array'],
        'video.frames_per_second' : 30
    }

    def __init__(self):
        self.reset(False)
        self.viewer = None

        #max_action = np.array([MAX_TORQUE1, MAX_TORQUE2])
        max_obs    = np.array([ np.pi/2 , np.pi/2 , np.pi/2 , MAX_ANGV , MAX_ANGV , MAX_ANGV ], dtype=np.float32)

        self.action_space = spaces.Discrete(4)
        #self.action_space = spaces.Box(low=-max_action, high=max_action, dtype=np.float32)
        self.observation_space = spaces.Box(low=-max_obs, high=max_obs, dtype=np.float32)

        self.seed()

        self.step_render= int(os.environ.get('RENDER', "0"))

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def stepWithU(self, u):

        _, t, s, _, _ = self.history[-1]

        # 1. update model
        mode, t, s = mp.step(t, s, u, delta)

        # 2. check done flag
        done = game_over(s)

        # 3. calc reward
        r = 11 - stage_cost(s, u)

        # 4. rendering
        if self.step_render != 0:
            self.render(-1)

        # 5. logging
        self.history.append((mode, t, s, u, r))

        return obs(s), r, done, {}

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
        return self.stepWithU(u)

    def reset_frame(self, frame):
        self.history = self.history[:frame+1]

    def reset(self, random=True):
        s = mp.reset_state(self.np_random if random else None)
        self.history = [("start", 0, s, np.array([0,0]), 0)]
        return obs(s)

    def render(self, mode='human', frame=-1):
        if self.viewer is None:
            self.viewer = RabbitViewer()
        mode, t, s, u, r = self.history[frame]
        return self.viewer.render(s, t, u, r)

    def frames(self):
        return len(self.history)

    def info(self, frame=-1):
        mode, t, s, u, r = self.history[frame]
        energy = mp.energy(s)
        print(f"--{t:.2f}-- E: {energy:.2f} --{mode:}-- reward: {r:}")
        print(f"u:  {u}")
        mp.print_state(s)
        print("")

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
        stepOne = False
        prev_frame = frame
        n = env.frames()
        for event in pygame.event.get():
            if event.type == pl.QUIT:
                env.close()
                sys.exit()
            elif event.type == pl.MOUSEBUTTONDOWN:
                start = start ^ True
            elif event.type == pl.KEYDOWN:
                keyname = pygame.key.name(event.key)
                mods = pygame.key.get_mods()

                # app
                if keyname == 'q':
                    env.close()
                    sys.exit()

                if keyname == 'r':
                    frame = 0
                    s = env.reset()
                    n = env.frames()

                elif keyname == 's':
                    start = start ^ True

                # frame rate
                elif keyname == 'd':
                    wait_rate = min(20, wait_rate + 1)
                elif keyname == 'u':
                    wait_rate = max(0, wait_rate - 1)

                elif keyname == 'space':
                    stepOne = True
                    start = True
                # input U
                elif keyname == 'h':
                    u = MAX_TORQUE1 * np.array([-1, 0])
                elif keyname == 'l':
                    u = MAX_TORQUE1 * np.array([1, 0])
                elif keyname == 'j':
                    u = MAX_TORQUE2 * np.array([0, -1])
                elif keyname == 'k':
                    u = MAX_TORQUE2 * np.array([0, 1])
                elif keyname == ';':
                    u = np.array([0, 0])

                # history
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

        env.render(frame=frame)

        if start:
            if stepOne:
                env.reset_frame(frame)
                _, _, done, _ = env.stepWithU(u)
                start = False
            elif frame == n-1:
                _, _, done, _ = env.stepWithU(u)
                if done:
                    start = False
            frame = frame + 1
            env.info(frame= -1 if frame == n-1 else frame)
        time.sleep(wait_rate * delta)

if __name__ == '__main__':
    main()
