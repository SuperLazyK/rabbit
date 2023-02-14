import pygame
import pygame.locals as pl
import pygame_menu
import numpy as np
import glob
from numpy import sin, cos
from math import degrees
from os import path
import os
import time
import control as ct
import sys
#import model_pogo_rot_limit as mp
import model_pogo_3point as mp
import csv
import datetime
import pickle
import yaml
from scipy import interpolate



pygame.init()
# input U
DELTA = 0.001
FRAME_RATE=30
#DELTA = 0.002
#DELTA = 0.005
SPEED=1000

NORMAL_MODE=0
JUMP_MODE=1
FLIP_MODE=2

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

def dump_history_csv(history, filename='state.csv'):
    pass

def dump_plot(d, filename='plot.csv'):
    pass

class RabbitViewer():
    def __init__(self):
        self.screen = pygame.display.set_mode(SCREEN_SIZE)
        pygame.display.set_caption("pogo-arm")
        self.clock = pygame.time.Clock()
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

    def render(self, state, mode, t, ref, u, r):

        energy = mp.energy(state)

        ps = list(mp.node_pos(state))
        for i in range(len(ps)):
            ps[i] = self.conv_pos(ps[i])
        cog = self.conv_pos(mp.cog(state))
        #head = self.conv_pos(mp.head_pos(state))

        self.screen.fill(WHITE)

        for i in range(1, len(ps)-2):
            pygame.draw.line(self.screen, BLACK, ps[i], ps[i+1],  width=int(100 * RSCALE))
        pygame.draw.line(self.screen, BLACK, ps[-2], ps[-1],  width=int(100 * RSCALE))
        pygame.draw.line(self.screen, GREEN, ps[1], ps[-1],  width=int(100 * RSCALE))
        pygame.draw.line(self.screen, GRAY, ps[0], ps[1],  width=int(100 * RSCALE))

        for i in range(len(ps)):
            pygame.draw.circle(self.screen, GRAY, ps[i], 150/5 * np.sqrt(RSCALE))
        #pygame.draw.circle(self.screen, RED, ps[0], 150/5 * np.sqrt(RSCALE))
        pygame.draw.circle(self.screen, BLUE, ps[4], 150/5 * np.sqrt(RSCALE))

        pygame.draw.circle(self.screen, RED, cog, 150/5 * np.sqrt(RSCALE))
        #pygame.draw.circle(self.screen, YELLOW, head, 150/5 * np.sqrt(RSCALE))
        pygame.draw.line(self.screen, BLACK, [0,SCREEN_SIZE[1]/2 + OFFSET_VERT], [SCREEN_SIZE[0], SCREEN_SIZE[1]/2 + OFFSET_VERT],  width=int(100 * RSCALE))
        #pygame.draw.line(self.screen, BLACK, ps[-2], head,  width=int(100 * RSCALE))
        tmx, tmy, am = mp.moment(state)
        text = self.font.render(f"mode={mode:} t={t:.03f} E={energy:.01f}", True, BLACK)
        text1 = self.font.render(f"ref={degrees(ref[0]):.01f} {(ref[1]):.02f}", True, BLACK)
        info = mp.calc_joint_property(state)
        text2 = self.font.render(f"obs={degrees(info['th0']):.01f} {(info['thk']):.02f}", True, BLACK)
        text3 = self.font.render(f"moment={tmx:.01f} {tmy:.02f} {am:.02f}", True, BLACK)
        self.screen.blit(text, [300, 50])
        self.screen.blit(text1, [300, 100])
        self.screen.blit(text2, [300, 150])
        self.screen.blit(text3, [300, 200])
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
        #self.alpha = 1
        #self.alpha = 0
        self.alpha = 0.5
        self.viewer = None
        self.is_render_enabled= int(os.environ.get('RENDER', "0"))

        self.min_action = mp.REF_MIN
        self.max_action = mp.REF_MAX
        self.min_obs = mp.OBS_MIN
        self.max_obs = mp.OBS_MAX

    def autosave(self, dirname='autodump'):
        pass

    def reset(self, random=None):
        if len(self.history ) > 1:
            if int(os.environ.get('AUTOSAVE', "0")):
                self.autosave("normal")

        th0 = np.deg2rad(0)
        pr = np.array([0, 0])
        thr =  0
        thk = np.deg2rad(2)

        s = mp.reset_state(pr, thr, th0, thk)
        self.mode = NORMAL_MODE
        t = 0
        u = (0, 0)
        reward = 0
        ref = mp.init_ref(s)
        self.history = [(self.mode, t, s, ref, u, reward)]

        return mp.obs(s)

    def step(self, act):
        s, reward, done, p = self.step_pos_control(act)
        return mp.obs(s), reward, done, p

    def obs(self, s):
        return mp.obs(s)

    def calc_reward(self, s, mode, t, done):
        if done:
            return 0
        if mode == JUMP_MODE:
            r = mp.reward_imitation_jump(s, t)
        else:
            r = mp.reward(s)
        return max(0, r)

    def num_of_frames(self):
        return len(self.history)

    def rollback(self, frame):
        self.history = self.history[:frame+1]

    def game_over(self, s):
        is_ok, reason = mp.check_invariant(s)
        return not is_ok, reason

    def step_plant(self, u, ref=np.array([0, 0])):
        _, t, s, _, _, _ = self.history[-1]
        success, t, s = mp.step(t, s, u, DELTA)
        if not success:
            self.autosave("failure")
            print("GAMEOVER : step-failure!!")
            done = True
        else:
            done, reason = self.game_over(s)
            if done:
                print(reason)
        reward = self.calc_reward(s, self.mode, t, done)
        self.history.append((self.mode, t, s, ref, u, reward))

        #if self.is_render_enabled != 0:
        #    self.render(-1)

        return s, reward, done, {}

    def step_pos_control(self, pos_ref):
        _, t, s, prev, _, _ = self.history[-1]
        t1 = t + 1.0/FRAME_RATE
        #t1 = t + DELTA #30Hz

        while t < t1:
            prev = (1-self.alpha) * prev + (self.alpha) * pos_ref
            u = mp.pdcontrol(s, prev)
            u = mp.torq_limit(s, u)
            s, reward, done, p = self.step_plant(u, prev)
            if done:
                break
            _, t, s, _, _, _ = self.history[-1]

        if self.is_render_enabled != 0:
            self.render(-1)

        return s, reward, done, p

    def step_vel_control(self, v_ref):
        _, t, s, ref, _, _ = self.history[-1]
        pos_ref = mp.ref_clip(ref + v_ref)
        return self.step_pos_control(pos_ref)

    def render(self, frame=-1):
        if self.viewer is None:
            self.viewer = RabbitViewer()
        mode, t, s, ref, u, reward = self.history[frame]
        return self.viewer.render(s, mode, t, ref, u, reward)

    def set_fixed_constraint_0t(self, frame=-1):
        mp.set_fixed_constraint_0t(self.history[frame][2])

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
        mode, _, _, ref, u, reward = self.history[frame]
        success, t, s = mp.step(t, prev_s, u, DELTA)
        if not success:
            print("failure!!")
        done, reason = self.game_over(s)
        reward = self.calc_reward(s, mode, t, done)


    def info(self, frame=-1):
        mode, t, s, ref, u, reward = self.history[frame]
        energy = mp.energy(s)
        print(f"--")
        print(f"t:{t:.3f} E: {energy:.2f} mode: {mode:} reward: {reward:}")
        print(f"INPUT: u: {100*(u/mp.max_u())} [%]")
        print(f"INPUT: ref: th0 {degrees(ref[0]):.2f}  thk {degrees(ref[1]):.2f}")
        print(f"OUTPUT:")
        self.joint_info(frame)
        #mp.print_state(s)
        print(f"--------------")

    def save(self, filename='dump.pkl'):
        pass

    def load(self, filename='dump.pkl'):
        pass

    def joint_info(self, frame):
        s = self.history[frame][2]
        prop = mp.calc_joint_property(s)
        prop['t'] = self.history[frame][1]
        for k in prop:
            if 'th' in k:
                prop[k] = degrees(prop[k])
            prop[k] = round(float(prop[k]), 3)
            print(f"{k}\t:\t{prop[k]}")
        return prop

    def load_plot(self, filename):
        pass

#----------------------------
# main
#----------------------------

def exec_cmd(env, v):
    #ctr_mode = 'torq'
    ctr_mode = 'vel'
    if ctr_mode == 'vel':
        k_th0 = SPEED/6*np.pi/360
        k_a   = SPEED/6*np.pi/360
        _, _, done, _ = env.step_vel_control(np.array([k_th0, k_a]) * v)
    else:
        k_th0 = 100000
        k_a   = 100000
        _, _, done, _ = env.step_plant(np.array([k_th0, k_a]) * v)
    return done

def fetch_episodes(dirname):
    files = glob.glob(dirname + "/*.pkl")
    files.sort(key=os.path.getmtime)
    return files



def main():

    pygame.init()

    env = RabbitEnv()

    v = np.array([0, 0])
    frame = env.num_of_frames() - 1
    last_frame = -1
    start = False
    pygame.event.clear()
    done = False
    replay = False
    slow = False
    episodes = []
    if len(sys.argv) > 1:
        for dirname in sys.argv[1:]:
            episodes = episodes + fetch_episodes(dirname)
        eidx = 0
        env.load(episodes[eidx])
        last_episode = -1
        done = True
        replay = True
    move_point_idx = None

    env.render(frame=0)
    plot_data = []

    while True:
        stepOne = False
        prev_frame = frame
        n = env.num_of_frames()
        for event in pygame.event.get():
            if event.type == pl.QUIT:
                env.close()
                sys.exit()
            elif event.type == pl.KEYDOWN:
                keyname = pygame.key.name(event.key)
                mods = pygame.key.get_mods()

                # app
                if keyname == 'w':
                    env.save()
                elif keyname == 'c':
                    env.set_fixed_constraint_0t(frame)
                elif keyname == '1':
                    print("load")
                    env.load('dump.pkl')
                    frame = 0
                    replay = True
                    done = True
                elif keyname == '0':
                    env.load_plot('main_pose.csv')
                    frame = 0
                    replay = True
                    done = True

                if keyname == 'q':
                    #dump_history_csv(env.history)
                    env.close()
                    sys.exit()

                if keyname == 'r':
                    frame = 0
                    done = False
                    replay = False
                    start = False
                    s = env.reset()
                    n = env.num_of_frames()
                    env.render(frame=0)

                elif keyname == 's':
                    start = start ^ True

                # frame rate
                elif keyname == 'd':
                    mp.debug = mp.debug ^ True

                elif keyname == 'i':
                    env.info(frame)
                    d = env.joint_info(frame)
                    #plot_data.append( {k: d[k] for k in ['t', 'z', 'prx', 'pry', 'thr', 'th0', 'thk']})
                    #dump_plot(plot_data)

                elif keyname == 'u':
                    slow = slow ^ True
                elif keyname == 'space':
                    stepOne = True
                    start = True
                    replay = False

                # history
                elif replay and keyname == 'n':
                    start = False
                    if mods & pl.KMOD_LSHIFT:
                        frame = frame + 30
                    else:
                        frame = frame + 1
                    if frame >= n:
                        frame = 0
                elif replay and keyname == 'p':
                    start = False
                    if mods & pl.KMOD_LSHIFT:
                        frame = frame - 30
                    else:
                        frame = frame - 1
                    if frame < 0:
                        frame = n-1
                elif replay and keyname == 'h':
                    start = False
                    if mods & pl.KMOD_LSHIFT:
                        eidx = max(eidx - 10, 0)
                    else:
                        eidx = max(eidx - 1, 0)
                elif replay and keyname == 'l':
                    start = False
                    if mods & pl.KMOD_LSHIFT:
                        eidx = min(eidx + 10, len(episodes)-1)
                    else:
                        eidx = min(eidx + 1, len(episodes)-1)
                # input
                elif keyname == 'j':
                    v = -np.array([0, -1])
                elif keyname == 'k':
                    v =  -np.array([0, 1])
                elif keyname == 'h':
                    v = np.array([1, 0])
                elif keyname == 'l':
                    v = np.array([-1, 0])
            elif event.type == pl.KEYUP:
                v = np.array([0, 0])

        if replay:
            if start:
                frame = frame + 1
                if frame > env.num_of_frames()-1:
                    frame = 0
                    if len(episodes) > 0:
                        eidx = eidx + 1
                        if eidx == len(episodes)-1:
                            eidx = 0

            if len(episodes) > 0 and last_episode != eidx:
                env.load(episodes[eidx])
                frame = 0
                last_episode = eidx

            if last_frame != frame:
                if (slow or not start) or (frame % 3 == 0):
                    env.render(frame=frame)
                    env.dryrun(frame)
                    if mp.debug:
                        env.info(frame)
                last_frame = frame

        elif start and not done:
            done = exec_cmd(env, v)
            frame = env.num_of_frames() - 1
            env.render(frame=frame)
            env.info()
            if stepOne:
                start = False
                stepOne = False
            if done:
                print("done!")
                start = False
                replay = True


if __name__ == '__main__':
    main()

