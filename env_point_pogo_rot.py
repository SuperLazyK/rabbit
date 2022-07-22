import pygame
import pygame.locals as pl
import pygame_menu
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
#import model_pogo_rot_limit as mp
import model_pogo_phy as mp
import csv
import pickle

# input U
#DELTA = 0.001
DELTA = 0.01
SPEED=6

#----------------------------
# Rendering
#----------------------------

FRAME_RATE = 30

BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
YELLOW = (158, 158, 30)
RED = (228, 0, 0)
GREEN = (0, 228, 0)
BLUE = (0, 0, 228)
GRAY = (100, 100, 100)
SCREEN_SIZE=(1300, 500)
SCALE=100
RSCALE=1/SCALE
OFFSET_VERT = SCREEN_SIZE[1]/3

class RabbitViewer():
    def __init__(self):
        self.screen = pygame.display.set_mode(SCREEN_SIZE)
        pygame.display.set_caption("rabbot-4-mass-point")
        self.clock = pygame.time.Clock()
        self.rotate = pygame.image.load("clockwise.png")
        self.font = pygame.font.SysFont('Calibri', 25, True, False)

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

    def flip(self, p):
        ret = p.copy()
        ret[1] = -ret[1]
        return ret

    def conv_pos(self, p):
        ret = self.flip(SCALE * p) + np.array(SCREEN_SIZE)/2 +  np.array([0, OFFSET_VERT])
        return ret

    def revert_pos(self, p):
        return self.flip((p - (np.array(SCREEN_SIZE)/2 +  np.array([0, OFFSET_VERT]))) * RSCALE)

    def get_point_idx(self, state, point, radius):
        ps = list(mp.node_pos(state))
        for i in range(len(ps)):
            ps[i] = self.conv_pos(ps[i])
            if np.linalg.norm(ps[i] - point) < radius:
                return i
        return None

    def set_fixed_constraint(self, idx, point):
        if point is not None:
            point = self.revert_pos(point)

        return mp.set_fixed_constraint(idx, point)

    def render(self, state, t, ref, u, r):

        energy = mp.energy(state)

        ps = list(mp.node_pos(state))
        for i in range(len(ps)):
            ps[i] = self.conv_pos(ps[i])
        cog = self.conv_pos(mp.cog(state))

        self.screen.fill(WHITE)
        for i in range(len(ps)-1):
            pygame.draw.line(self.screen, BLACK, ps[i], ps[i+1],  width=int(100 * RSCALE))
        if len(ps) == 5:
            pygame.draw.line(self.screen, BLACK, ps[1], ps[3],  width=int(100 * RSCALE))
        for i in range(len(ps)):
            pygame.draw.circle(self.screen, BLUE if i > 0 else RED, ps[i], 150/5 * np.sqrt(RSCALE))

        pygame.draw.circle(self.screen, GREEN, cog, 150/5 * np.sqrt(RSCALE))
        pygame.draw.line(self.screen, BLACK, [0,SCREEN_SIZE[1]/2 + OFFSET_VERT], [SCREEN_SIZE[0], SCREEN_SIZE[1]/2 + OFFSET_VERT])
        tmx, tmy, am = mp.moment(state)
        #text = self.font.render("t={:.02f} r={:.02f} ".format(t, energy, r), True, BLACK)
        text = self.font.render("t={:.03f} E={:.02f} AM={:.00f} TMx={:.00f} TMy={:.00f}".format(t, energy, am, tmx, tmy), True, BLACK)
        self.screen.blit(text, [300, 50])
        pygame.display.flip()
        self.clock.tick(60)

    def close(self):
        pygame.quit()

#--------------------------------
# Simulation Controller + Reward
#--------------------------------
class RabbitEnv():

    def __init__(self, seed=None):
        self.reset(False)
        self.viewer = None
        self.np_random , seed = seeding.np_random(seed)
        self.is_render_enabled= int(os.environ.get('RENDER', "0"))

    def num_of_frames(self):
        return len(self.history)

    def reset(self, random=True):
        s = mp.reset_state(self.np_random if random else None)
        ref = mp.init_ref(s)
        self.history = [("start", 0, s, ref, (0,0), 0)]
        return s

    def rollback(self, frame):
        self.history = self.history[:frame+1]

    def game_over(self, s):
        return not mp.check_invariant(s)

    def calc_reward(self, s):
        return 0

    def step_plant(self, u, ref=np.array([0, 0])):
        _, t, s, _, _, _ = self.history[-1]

        u = mp.torq_limit(s, u)
        t1 = t + DELTA

        # 1. update model
        while t < t1:
            mode, t, s = mp.step(t, s, u, DELTA)
            done = self.game_over(s)
            reward = self.calc_reward(s)
            self.history.append((mode, t, s, ref, u, reward))
            if done:
                break

        # 2. rendering
        if self.is_render_enabled != 0:
            self.render(-1)

        return s, reward, done, {}

    def step_pos_control(self, pos_ref):
        _, _, s, _, _, _ = self.history[-1]
        u = mp.pdcontrol(s, pos_ref)
        return self.step_plant(u, pos_ref)

    def step_vel_control(self, v_ref):
        _, t, s, ref, _, _ = self.history[-1]
        pos_ref = mp.ref_clip(ref + v_ref)
        return self.step_pos_control(pos_ref)

    def render(self, mode='human', frame=-1):
        if self.viewer is None:
            self.viewer = RabbitViewer()
        mode, t, s, ref, u, reward = self.history[frame]
        return self.viewer.render(s, t, ref, u, reward)

    def get_point_idx(self, point, radius, frame=-1):
        mode, t, s, ref, u, reward = self.history[frame]
        return self.viewer.get_point_idx(s, point, radius)

    def set_fixed_constraint(self, idx, point):
        return self.viewer.set_fixed_constraint(idx,  point)

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None

    #------------------------------------------
    # Debug functions
    #------------------------------------------
    def info(self, frame=-1):
        mode, t, s, ref, u, reward = self.history[frame]
        energy = mp.energy(s)
        print(f"--")
        print(f"t:{t:.3f} E: {energy:.2f} mode: {mode:} reward: {reward:}")
        print(f"INPUT: u: {u/mp.max_u()}")
        print(f"OUTPUT:")
        mp.print_state(s)
        print(f"--------------")

    def save(self):
        with open(r'dump.pkl','wb') as f:
            pickle.dump(self.history, f, pickle.HIGHEST_PROTOCOL)

    def load(self):
        with open(r'dump.pkl','rb') as f:
            self.history = pickle.load(f)
            self.history.pop(-1)

    def dump_csv(self):
        data = []
        for mode, t, s, ref, u, reward in self.history:
            energy = mp.energy(s)
            dic = {}
            dic['t'] = t
            dic['ref_th0'] = ref[0]
            dic['ref_th1'] = ref[1]
            dic['u_torq0%'] = u[0]/mp.max_u()[0]
            dic['u_torq1%'] = u[1]/mp.max_u()[1]
            dic = mp.state_dict(s, dic)
            dic['E'] = energy
            data.append(dic)

        with open(r'state.csv','w',encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames = dic.keys())
            writer.writeheader()
            writer.writerows(data)


#----------------------------
# main
#----------------------------

def exec_cmd(env, v):
    #ctr_mode = 'torq'
    ctr_mode = 'vel'
    if ctr_mode == 'vel':
        k_th0 = SPEED*np.pi/360
        k_th1 = SPEED*np.pi/360
        _, _, done, _ = env.step_vel_control(np.array([k_th0, k_th1]) * v)
    else:
        k_th0 = 100000
        k_th1 = 100000
        _, _, done, _ = env.step_plant(np.array([k_th0, k_th1]) * v)
    return done


def main():

    pygame.init()

    env = RabbitEnv()

    v = np.array([0, 0])
    wait_rate = 0
    frame = env.num_of_frames() - 1
    last_frame = -1
    start = False
    pygame.event.clear()
    done = False

    ref_th0 = 0
    replay = False
    move_point_idx = None

    while True:
        stepOne = False
        prev_frame = frame
        n = env.num_of_frames()
        for event in pygame.event.get():
            if event.type == pl.QUIT:
                env.close()
                sys.exit()
            elif event.type == pl.MOUSEBUTTONDOWN:
                move_point_idx = env.get_point_idx(pygame.mouse.get_pos(), 10, frame)
            elif event.type == pl.MOUSEBUTTONUP:
                env.set_fixed_constraint(move_point_idx, None)
                move_point_idx = None
            elif event.type == pl.MOUSEMOTION:
                if move_point_idx is not None:
                    env.set_fixed_constraint(move_point_idx, pygame.mouse.get_pos())
            elif event.type == pl.KEYDOWN:
                keyname = pygame.key.name(event.key)
                mods = pygame.key.get_mods()

                # app
                if keyname == 'w':
                    env.save()
                elif keyname == '1':
                    env.load()
                    frame = 0
                    replay = True
                    done = True

                if keyname == 'q':
                    env.dump_csv()
                    env.close()
                    sys.exit()

                if keyname == 'r':
                    frame = 0
                    done = False
                    s = env.reset()
                    n = env.num_of_frames()

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

                # input
                elif keyname == 'j':
                    v = np.array([1, 0])
                elif keyname == 'k':
                    v = np.array([-1, 0])
                elif keyname == 'h':
                    v =  np.array([0, 1])
                elif keyname == 'l':
                    v = np.array([0, -1])
            elif event.type == pl.KEYUP:
                v = np.array([0, 0])

        env.render(frame=frame)

        if start and replay:
            frame = frame + 1
            if frame == n-1:
                frame = 0
        elif start and not done:
            if stepOne:
                env.rollback(frame)
                done = exec_cmd(env, v)
                start = False
            elif frame == n-1:
                #_, _, done, _ = env.step_vel_control(v)
                done = exec_cmd(env, v)
                if done:
                    start = False
            frame = frame + 1

        if last_frame != frame:
            #env.info(frame= -1 if frame == n-1 else frame)
            last_frame = frame

        time.sleep(wait_rate * 1.0/FRAME_RATE)

if __name__ == '__main__':
    main()
