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
import model_pogo_phy as mp
import csv
import datetime
import pickle
import yaml
from scipy import interpolate



pygame.init()
# input U
DELTA = 0.001
#DELTA = 0.01
SPEED=6

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
    data = []
    for mode, t, s, ref, u, reward in history:
        energy = mp.energy(s)
        dic = {}
        dic['t'] = t
        dic['ref_th0'] = ref[0]
        dic['ref_th1'] = ref[1]
        dic['ref_d']   = ref[2]
        dic['u_torq0%'] = u[0]/mp.max_u()[0]
        dic['u_torq1%'] = u[1]/mp.max_u()[1]
        data.append(dic)

    with open(filename,'w',encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames = dic.keys())
        writer.writeheader()
        writer.writerows(data)

def dump_plot_yaml(joint_info, filename='plot.yaml'):
    with open(filename, 'w') as f:
        yaml.dump(joint_info, f)

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
        head = ps[-2]

        self.screen.fill(WHITE)
        for i in range(len(ps)-2):
            pygame.draw.line(self.screen, BLACK, ps[i], ps[i+1],  width=int(100 * RSCALE))
        pygame.draw.line(self.screen, BLACK, ps[-3], ps[-1],  width=int(100 * RSCALE))
        pygame.draw.line(self.screen, BLACK, ps[1], ps[-1],  width=int(100 * RSCALE))

        for i in range(len(ps)):
            pygame.draw.circle(self.screen, BLUE if i > 0 else RED, ps[i], 150/5 * np.sqrt(RSCALE))

        pygame.draw.circle(self.screen, GREEN, cog, 150/5 * np.sqrt(RSCALE))
        pygame.draw.circle(self.screen, YELLOW, head, 150/5 * np.sqrt(RSCALE))
        pygame.draw.line(self.screen, BLACK, [0,SCREEN_SIZE[1]/2 + OFFSET_VERT], [SCREEN_SIZE[0], SCREEN_SIZE[1]/2 + OFFSET_VERT],  width=int(100 * RSCALE))
        tmx, tmy, am = mp.moment(state)
        text = self.font.render(f"mode={mode:} t={t:.03f} E={energy:.01f}", True, BLACK)
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
        print("dump!!!")
        os.makedirs(dirname, exist_ok=True)
        self.save(dirname + '/last_episode.pkl')
        dt = datetime.datetime.now()
        timestamp = dt.strftime("%Y-%m-%d-%H-%M-%S")
        self.save(dirname + '/{}.pkl'.format(timestamp))

    def reset(self, random=None):
        if len(self.history ) > 1:
            if int(os.environ.get('AUTOSAVE', "0")):
                self.autosave("normal")

        th0 = np.deg2rad(10)
        thk = np.deg2rad(-20)
        th1 = np.deg2rad(15)
        if random is None:
            pr = np.array([0, 1])
            thr =  0
            vr = np.array([0, 0])
            dthr = 0
        else:
            if True: #random.randint(0, 2) == 0:
                pr = np.array([0, 0])
                thr = np.deg2rad(-45)
                vr = np.array([0, 0])
                dthr = 2.7*np.deg2rad(45)
            else:
                pr = np.array([0, 1])
                thr =  np.deg2rad(30) * np.random.rand()
                vr = np.array([5*np.random.rand(), 0])
                dthr = np.random.rand()

        s = mp.reset_state(pr, thr, th0, thk, th1, vr, dthr)
        self.mode = NORMAL_MODE
        t = 0
        u = (0, 0, 0)
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

    def step_plant(self, u, ref=np.array([0, 0, 0])):
        _, t, s, _, _, _ = self.history[-1]
        success, t, s = mp.step(t, s, u, DELTA)
        if not success:
            self.autosave("failure")
            print("failure!!")
            done = True
        else:
            done, reason = self.game_over(s)
        reward = self.calc_reward(s, self.mode, t, done)
        self.history.append((self.mode, t, s, ref, u, reward))

        #if self.is_render_enabled != 0:
        #    self.render(-1)

        return s, reward, done, {}

    def step_pos_control(self, pos_ref):
        _, t, s, prev, _, _ = self.history[-1]
        t1 = t + 0.033 #30Hz

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
        print(f"INPUT: u: {u/mp.max_u()}")
        print(f"OUTPUT:")
        mp.print_state(s)
        print(f"--------------")

    def save(self, filename='dump.pkl'):
        with open(filename,'wb') as f:
            pickle.dump(self.history, f, pickle.HIGHEST_PROTOCOL)

    def load(self, filename='dump.pkl'):
        print(f"Load {filename}")
        with open(filename,'rb') as f:
            self.history = pickle.load(f)
            #self.history.pop(-1)

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

    def load_yaml(self, filename):

        self.history = []
        with open(filename) as file:
            jprops = yaml.safe_load(file)

        tsi = np.arange(jprops[0]['t'], jprops[-1]['t'], DELTA)
        data = {}
        for field in ['z', 'prx', 'pry', 'thr', 'th0', 'thk', 'th1']:
            # duplicate t=0 with velocity 0
            ts = [jprops[0]['t']-1]
            xs = [jprops[0][field]]
            for jprop in jprops:
                ts.append(jprop['t'])
                xs.append(jprop[field])
            data[field] = interpolate.interp1d(ts, xs, kind="cubic")(tsi)

        for i in range(len(tsi)):
            t = tsi[i]
            if i == 0:
                dz = 0
                vrx = 0
                vry = 0
                dthr = 0
                dth0 = 0
                dthk = 0
                dth1 = 0
            else:
                dz   = (data['z']  [i]  - data['z']  [i-1]) / DELTA
                vrx  = (data['prx'][i]  - data['prx'][i-1]) / DELTA
                vry  = (data['pry'][i]  - data['pry'][i-1]) / DELTA
                dthr = (data['thr'][i]  - data['thr'][i-1]) / DELTA
                dth0 = (data['th0'][i]  - data['th0'][i-1]) / DELTA
                dthk = (data['thk'][i]  - data['thk'][i-1]) / DELTA
                dth1 = (data['th1'][i]  - data['th1'][i-1]) / DELTA
            s = mp.reset_state(
                    np.array([data['prx'][i], data['pry'][i]]),
                    np.deg2rad(data['thr'][i]),
                    np.deg2rad(data['th0'][i]),
                    np.deg2rad(data['thk'][i]),
                    np.deg2rad(data['th1'][i]),
                    np.array([vrx, vry]),
                    np.deg2rad(dthr),
                    np.deg2rad(dth0),
                    np.deg2rad(dthk),
                    np.deg2rad(dth1),
                    data['z'][i],
                    dz,
                    )
            self.mode = NORMAL_MODE
            u = (0, 0, 0)
            mode ='normal'
            ref = mp.init_ref(s)
            reward = self.calc_reward(s, mode, t, False)
            self.history.append((self.mode, t, s, ref, u, reward))

#----------------------------
# main
#----------------------------

def exec_cmd(env, v):
    #ctr_mode = 'torq'
    ctr_mode = 'vel'
    if ctr_mode == 'vel':
        k_th0 = SPEED*np.pi/360
        k_th1 = SPEED*np.pi/360
        k_a   = SPEED/6 * 0.01
        _, _, done, _ = env.step_vel_control(np.array([k_th0, k_th1, k_a]) * v)
    else:
        k_th0 = 100000
        k_th1 = 100000
        k_a   = 100000
        _, _, done, _ = env.step_plant(np.array([k_th0, k_th1, k_a]) * v)
    return done

def fetch_episodes(dirname):
    files = glob.glob(dirname + "/*.pkl")
    files.sort(key=os.path.getmtime)
    return files



def main():

    pygame.init()

    env = RabbitEnv()

    v = np.array([0, 0, 0])
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
                elif keyname == '0':
                    env.load_yaml('main_pose.yaml')
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
                    s = env.reset()
                    n = env.num_of_frames()

                elif keyname == 's':
                    start = start ^ True

                # frame rate
                elif keyname == 'd':
                    mp.debug = mp.debug ^ True

                elif keyname == 'i':
                    env.info(frame)
                    plot_data.append(env.joint_info(frame))
                    dump_plot_yaml(plot_data, 'plot.yaml')

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
                        frame = frame + 10
                    else:
                        frame = frame + 1
                    if frame >= n:
                        frame = 0
                elif replay and keyname == 'p':
                    start = False
                    if mods & pl.KMOD_LSHIFT:
                        frame = frame - 10
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
                    v = np.array([-1, 0, 0])
                elif keyname == 'k':
                    v = np.array([1, 0, 0])
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
                if frame == env.num_of_frames()-1:
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
            if stepOne:
                start = False
                stepOne = False
            if done:
                print("done!")
                start = False
                replay = True
            #env.info()


if __name__ == '__main__':
    main()

