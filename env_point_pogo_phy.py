import pygame
import pygame.locals as pl
import pygame_menu
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
import datetime
import pickle


pygame.init()
# input U
#DELTA = 0.001
DELTA = 0.01
SPEED=6

JUMP_MODE=0
FLIP_MODE=1

#----------------------------
# Rendering
#----------------------------

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
        pygame.display.set_caption("pogo-arm")
        self.clock = pygame.time.Clock()
        #self.rotate = pygame.image.load("clockwise.png")
        self.font = pygame.font.SysFont('Calibri', 25, True, False)

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

    def render(self, state, mode, t, ref, u, r):

        energy = mp.energy(state)

        ps = list(mp.node_pos(state))
        for i in range(len(ps)):
            ps[i] = self.conv_pos(ps[i])
        cog = self.conv_pos(mp.cog(state))
        head = self.conv_pos(mp.head_pos(state))

        self.screen.fill(WHITE)
        for i in range(len(ps)-1):
            pygame.draw.line(self.screen, BLACK, ps[i], ps[i+1],  width=int(100 * RSCALE))
        pygame.draw.line(self.screen, BLACK, ps[1], ps[-1],  width=int(100 * RSCALE))
        for i in range(len(ps)):
            pygame.draw.circle(self.screen, BLUE if i > 0 else RED, ps[i], 150/5 * np.sqrt(RSCALE))

        pygame.draw.circle(self.screen, GREEN, cog, 150/5 * np.sqrt(RSCALE))
        pygame.draw.circle(self.screen, YELLOW, head, 150/5 * np.sqrt(RSCALE))
        pygame.draw.line(self.screen, BLACK, ps[-2], head,  width=int(100 * RSCALE))
        pygame.draw.line(self.screen, BLACK, [0,SCREEN_SIZE[1]/2 + OFFSET_VERT], [SCREEN_SIZE[0], SCREEN_SIZE[1]/2 + OFFSET_VERT])
        tmx, tmy, am = mp.moment(state)
        text = self.font.render(f"mode={mode:} t={t:.03f}", True, BLACK)
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
        self.history = []
        self.reset()
        self.viewer = None
        self.is_render_enabled= int(os.environ.get('RENDER', "0"))

        self.min_action = mp.REF_MIN
        self.max_action = mp.REF_MAX
        self.min_obs = mp.OBS_MIN
        self.max_obs = mp.OBS_MAX

    def reset(self, random=None):
        print("RESET: NOT IMPLEMENTED")
        if len(self.history ) > 1:
            if int(os.environ.get('AUTOSAVE', "1")):
                os.makedirs('autodump', exist_ok=True)
                self.save('autodump/last_episode.pkl')
                self.dump_csv('autodump/last_episode.csv')
                if int(os.environ.get('USE_TIMESTAMP', "1")):
                    dt = datetime.datetime.now()
                    timestamp = dt.strftime("%Y-%m-%d-%H-%M-%S")
                    self.save('autodump/{}.pkl'.format(timestamp))
                    self.dump_csv('autodump/{}.csv'.format(timestamp))
        mode = JUMP_MODE
        t = 0
        act = (0, 0, 0)
        reward = 0
        s = mp.reset_state()
        ref = mp.init_ref(s)
        self.history = [(mode, t, s, ref, act, reward)]
        return mp.obs(s)

    def step(self, act):
        s, reward, done, p = self.step_pos_control(act)
        return mp.obs(s), reward, done, p

    def obs(self, s):
        return mp.obs(s)

    def calc_reward(self, s):
        print("REWARD: NOT IMPLEMENTED")
        return 0

    def num_of_frames(self):
        return len(self.history)

    def rollback(self, frame):
        self.history = self.history[:frame+1]

    def game_over(self, s):
        return not mp.check_invariant(s)

    def step_plant(self, u, ref=np.array([0, 0, 0])):
        _, t, s, _, _, _ = self.history[-1]

        u = mp.torq_limit(s, u)
        t1 = t + 3*DELTA #30msec

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

    def render(self, frame=-1):
        if self.viewer is None:
            self.viewer = RabbitViewer()
        mode, t, s, ref, u, reward = self.history[frame]
        return self.viewer.render(s, mode, t, ref, u, reward)

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
    def dryrun(self, frame):
        if frame == 0:
            return
        _, t, prev_s, _, _, _ = self.history[frame-1]
        _, _, _, ref, u, reward = self.history[frame]
        mode, t, s = mp.step(t, prev_s, u, DELTA)
        done = self.game_over(s)
        reward = self.calc_reward(s)


    def info(self, frame=-1):
        mode, t, s, ref, u, reward = self.history[frame]
        energy = mp.energy(s)
        print(f"--")
        print(f"t:{t:.3f} E: {energy:.2f} mode: {mode:} reward: {reward:}")
        print(f"INPUT: u: {u/mp.max_u()}")
        print(f"OUTPUT:")
        mp.print_state(s)
        print(f"--------------")

    def save(self, filename='dump.pkl'):
        with open(filename,'wb') as f:
            pickle.dump(self.history, f, pickle.HIGHEST_PROTOCOL)

    def load(self, filename='dump.pkl'):
        with open(filename,'rb') as f:
            self.history = pickle.load(f)
            self.history.pop(-1)

    def dump_csv(self, filename='sate.csv'):
        data = []
        for mode, t, s, ref, u, reward in self.history:
            energy = mp.energy(s)
            dic = {}
            dic['t'] = t
            dic['ref_th0'] = ref[0]
            dic['ref_th1'] = ref[1]
            dic['ref_d']   = ref[2]
            dic['u_torq0%'] = u[0]/mp.max_u()[0]
            dic['u_torq1%'] = u[1]/mp.max_u()[1]
            dic['E'] = energy
            data.append(dic)

        with open(filename,'w',encoding='utf-8') as f:
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
        k_a   = 0.2
        _, _, done, _ = env.step_vel_control(np.array([k_th0, k_th1, k_a]) * v)
    else:
        k_th0 = 100000
        k_th1 = 100000
        k_a   = 100000
        _, _, done, _ = env.step_plant(np.array([k_th0, k_th1, k_a]) * v)
    return done


def main():

    pygame.init()

    env = RabbitEnv()

    v = np.array([0, 0, 0])
    wait_rate = 0
    frame = env.num_of_frames() - 1
    last_frame = -1
    start = False
    pygame.event.clear()
    done = False

    replay = False
    move_point_idx = None

    env.is_render_enabled= 1
    env.render(frame=0)

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
                    print("load")
                    env.load('autodump/last_episode.pkl')
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
                    replay = False

                # history
                elif replay and keyname == 'n':
                    start = False
                    if mods & pl.KMOD_LSHIFT:
                        frame = min(frame + 10, n-1)
                    else:
                        frame = min(frame + 1, n-1)
                elif replay and keyname == 'p':
                    start = False
                    if mods & pl.KMOD_LSHIFT:
                        frame = max(frame - 10, 0)
                    else:
                        frame = max(frame - 1, 0)
                # input
                elif keyname == 'j':
                    v = np.array([1, 0, 0])
                elif keyname == 'k':
                    v = np.array([-1, 0, 0])
                elif keyname == 'h':
                    v =  np.array([0, 1, 0])
                elif keyname == 'l':
                    v = np.array([0, -1, 0])
                elif keyname == 'n':
                    v =  np.array([0, 0, 1])
                elif keyname == 'p':
                    v = np.array([0, 0, -1])
            elif event.type == pl.KEYUP:
                v = np.array([0, 0, 0])

        if replay:
            if start:
                frame = frame + 1
                if frame == n-1:
                    frame = 0
            if stepOne:
                env.rollback(frame)
                done = exec_cmd(env, v)
                start = False
                replay = False
            if last_frame != frame:
                env.render(frame=frame)
                env.dryrun(frame)
                env.info(frame)
                last_frame = frame
        elif start and not done:
            done = exec_cmd(env, v)
            if done:
                print("done!")
                start = False
            env.info()

if __name__ == '__main__':
    main()
