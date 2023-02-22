import pygame
import pygame.locals as pl
import pygame_menu
import numpy as np
import glob
from numpy import sin, cos
from math import degrees, radians
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


use_polar = True

pygame.init()
# input U
DELTA = 0.002
#DELTA = 0.001
FRAME_RATE=30
#FRAME_RATE=1000
#DELTA = 0.002
#DELTA = 0.005
#SPEED=1000
#SPEED=100
SPEED=30

NORMAL_MODE=0
JUMP_MODE=1
FLIP_MODE=2
CSV_FIELDS=['t', 'prx', 'pry', 'thr', 'z', 'th0', 'thk', 'thw']

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
        dic['ref_thk'] = ref[1]
        dic['ref_thw'] = ref[2]
        dic['u_torq0%'] = u[0]/mp.max_u()[0]
        dic['u_torqk%'] = u[1]/mp.max_u()[1]
        dic['u_torqw%'] = u[2]/mp.max_u()[2]
        dic = mp.calc_joint_property(s, dic)
        data.append(dic)

    with open(filename,'w',encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames = dic.keys())
        writer.writeheader()
        writer.writerows(data)

def dump_plot(history, filename='plot.csv'):
    d = []
    for _, t, s, _, _, _ in history:
        prop = {}
        joint = mp.calc_joint_property(s)
        prop['t'] = t
        for k in CSV_FIELDS[1:]:
            prop[k] = joint[k]
            if 'th' in k:
                prop[k] = degrees(prop[k])
            prop[k] = round(float(prop[k]), 3)
        cog = mp.cog(s)
        prop['cogx'] = cog[0]
        prop['cogy'] = cog[1]
        d.append(prop)

    with open(filename,'w',encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames = CSV_FIELDS+['cogx', 'cogy'])
        writer.writeheader()
        writer.writerows(d)

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
        text1 = self.font.render(f"ref={degrees(ref[0]):.01f} {degrees(ref[1]):.02f} {degrees(ref[2]):.02f}", True, BLACK)
        info = mp.calc_joint_property(state)
        text2 = self.font.render(f"obs={degrees(info['th0']):.01f} {degrees(info['thk']):.02f} {degrees(info['thw']):.02f}", True, BLACK)
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

        # use only shape!
        self.min_action = np.zeros(mp.DIM_U)
        self.max_action = np.zeros(mp.DIM_U)
        self.min_obs = mp.OBS_MIN
        self.max_obs = mp.OBS_MAX
        self.reference = self.load_plot('main_pose.csv')

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
        if random is not None:
            i = random.randint(len(self.reference))
            s = self.reference[i][2]
            done, msg = self.game_over(s)
            if done:
                print(i, "/", len(self.reference))
                print(s)
                assert not done, "???before-start???" + msg
        else:
            s = mp.reset_state({'pry': 1.2})
            #s = mp.reset_state({
            # 'prx': 0.00,
            # 'pry': 0.00,
            # 'thr': 0.00,
            # 'z': -0.01,
            # 'th0': -0.18,
            # 'thk': 0.46,
            # 'thw': -0.08,
            # 'dprx': 0.00,
            # 'dpry': 0.00,
            # 'dthr': 0.00,
            # 'dz': 7.33,
            # 'dth0': 1.30,
            # 'dthk': -0.07,
            # 'dthw': 2.65})
            done, msg = self.game_over(s)
            assert not done, "???before-start???" + msg
        self.mode = NORMAL_MODE
        t = 0
        u = mp.DEFAULT_U
        reward = 0
        ref = mp.init_ref(s)
        self.history = [(self.mode, t, s, ref, u, reward)]

        return mp.obs(s)

    def step(self, act):
        print("ACT=", act)
        if use_polar:
            #scaled = act*(mp.PREF_MAX - mp.PREF_MIN)/2 + (mp.PREF_MAX + mp.PREF_MIN)/2
            #scaled = act*(mp.PREF_MAX - mp.PREF_MIN) + (mp.PREF_MAX + mp.PREF_MIN)/2
            #ref = mp.from_polar(mp.pref_clip(scaled))
            ref = mp.from_polar(mp.pref_clip(act))
            s, reward, done, p = self.step_pos_control(ref)
        else:
            ref = mp.ref_clip(act)
            s, reward, done, p = self.step_pos_control(ref)
        return mp.obs(s), reward, done, p

    def obs(self, s):
        return mp.obs(s)

    def calc_reward(self, s, mode, t, done):
        if done:
            return 0.1
        r = mp.reward(s)
        return max(0, r)

    def num_of_frames(self):
        return len(self.history)

    def rollback(self, frame):
        self.history = self.history[:frame+1]

    def game_over(self, s):
        is_ok, reason = mp.check_invariant(s)
        return not is_ok, reason

    def step_plant(self, u, ref=mp.DEFAULT_U):
        _, t, s, _, _, _ = self.history[-1]
        success, mode, t, s = mp.step(t, s, u, DELTA)
        done, reason = self.game_over(s)
        if done:
            print(reason)
            pass
        elif not success:
            print("failure")
            done = True
        reward = self.calc_reward(s, self.mode, t, done)
        self.history.append((mode, t, s, ref, u, reward))

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
            s, reward, done, p = self.step_plant(u, prev)
            if done:
                break
            _, t, s, _, _, _ = self.history[-1]

        if self.is_render_enabled != 0:
            self.render(-1)

        return s, reward, done, p

    def step_vel_control(self, v_ref):
        _, t, s, ref, _, _ = self.history[-1]
        #pos_ref = mp.ref_clip_scale(ref, v_ref)
        pos_ref = mp.ref_clip(ref + v_ref)
        return self.step_pos_control(pos_ref)

    def step_pvel_control(self, v_ref):
        _, t, s, ref, _, _ = self.history[-1]
        prev_pref = mp.to_polar(ref)
        new_pref = mp.pref_clip(prev_pref + v_ref)
        pos_ref = mp.from_polar(new_pref)
        #pos_ref = mp.pref_clip(pos_ref)
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
        _, _, _, ref, u, reward = self.history[frame]
        success, mode, t, s = mp.step(t, prev_s, u, DELTA)
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

    def load_plot(self, filename):
        with open(filename) as f:
            reader = csv.DictReader(f)
            jprops = [{field:radians(float(row[field])) if 'th' in field else float(row[field]) for field in row} for row in reader]

        tsi = np.arange(jprops[0]['t'], jprops[-1]['t'], DELTA)
        data = {}
        for field in CSV_FIELDS[1:]:
            # duplicate t=0 with velocity 0
            ts = [jprops[0]['t']-1]
            xs = [jprops[0][field]]
            for jprop in jprops:
                ts.append(jprop['t'])
                xs.append(jprop[field])
            data[field] = interpolate.interp1d(ts, xs, kind="quadratic")(tsi)

        if True: # thr = 0
            ts = [jprops[0]['t']-1]
            cog = mp.cog(mp.reset_state((jprops[0])))
            cogxs = [cog[0]]
            cogys = [cog[1]]
            for jprop in jprops: # num of plots
                ts.append(jprop['t'])
                cog = mp.cog(mp.reset_state((jprop)))
                cogxs.append(cog[0])
                cogys.append(cog[1])
            cogx0 = interpolate.interp1d(ts, cogxs, kind="quadratic")(tsi)
            cogy0 = interpolate.interp1d(ts, cogys, kind="quadratic")(tsi)
            for i in range(tsi.shape[0]): # num of timestep
                d = {k:data[k][i] for k in data}
                s = mp.reset_state(d)
                z, prx, pry = mp.adjust_cog(np.array([cogx0[i], cogy0[i]]), s)
                data['z'][i] = z
                data['prx'][i] = prx
                data['pry'][i] = pry
                data['thr'][i] = mp.normalize_angle(data['thr'][i])

        history = []
        for i in range(len(tsi)):
            t = tsi[i]
            d = {}
            for field in CSV_FIELDS[1:]:
                d[field] = data[field][i]
                if i == 0:
                    d['d'+field] = 0
                else:
                    d['d'+field] = (data[field][i]  - data[field]  [i-1]) / DELTA
            s = mp.reset_state(d)
            self.mode = NORMAL_MODE
            mode ='normal'
            ref = mp.init_ref(s)
            reward = 0
            history.append((self.mode, t, s, ref, mp.DEFAULT_U, reward))

        return history

#----------------------------
# main
#----------------------------

def exec_cmd(env, v, frame):
    #ctr_mode = 'torq'
    #ctr_mode = 'vel'
    ctr_mode = 'polar'
    #ctr_mode = 'inv3'
    if ctr_mode == 'vel':
        k_th0 = 3*SPEED/6*np.pi/360
        k_thk = 3*SPEED/6*np.pi/360
        k_thw = 3*SPEED/6*np.pi/360
        _, _, done, _ = env.step_vel_control(np.array([k_th0, k_thk, k_thw]) * np.array([v[2], v[1], -v[0]]))
    elif ctr_mode == 'polar':
        k_th0 = 3*SPEED/6*np.pi/360
        k_thk = 3*SPEED/6*np.pi/360
        k_thw = 3*SPEED/6*np.pi/360
        _, _, done, _ = env.step_pvel_control(np.array([k_th0, k_thk, k_thw]) * np.array([v[2], v[1], -v[0]]))
    elif ctr_mode == 'inv3':
        k_th0 = 100*SPEED/6*np.pi/360
        k_r = SPEED/2000
        k_th = 10*SPEED/60*np.pi/360
        _, _, s, _, _, _ = env.history[frame]
        vj = mp.invkinematics3(s, np.array([0, -k_r, k_th]) * np.array([0, v[1], v[0]]))
        _, _, done, _ = env.step_vel_control(vj + np.array([k_th0*v[2], 0, 0]))
    else:
        k_th0 = 100000
        k_thk = 100000
        k_thw = 100000
        _, _, done, _ = env.step_plant(np.array([k_th0, k_thk, k_thw]) * v)
    return done

def fetch_episodes(dirname):
    files = glob.glob(dirname + "/*.pkl")
    files.sort(key=os.path.getmtime)
    return files



def main():

    pygame.init()

    env = RabbitEnv()

    v = mp.DEFAULT_U
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
    plot_data = [env.history[0]]

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
                    env.history = env.load_plot('main_pose.csv')
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
                    plot_data.append(env.history[0])

                elif keyname == 's':
                    start = start ^ True

                # frame rate
                elif keyname == 'd':
                    mp.debug = mp.debug ^ True

                elif keyname == 'i':
                    env.info(frame)

                elif keyname == 'u':
                    slow = slow ^ True
                    global FRAME_RATE
                    if slow:
                        FRAME_RATE = 1000
                    else:
                        FRAME_RATE = 30
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
                # input
                elif keyname == 'j':
                    v = -np.array([0, -1, 0])
                elif keyname == 'k':
                    v =  -np.array([0, 1, 0])
                elif keyname == 'h':
                    v = np.array([-1, 0, 0])
                elif keyname == 'l':
                    v = np.array([1, 0, 0])
                elif keyname == 'n':
                    v = np.array([0, 0, 1])
                elif keyname == 'p':
                    v = np.array([0, 0, -1])
                elif keyname == 'g':
                    if mp.g == 0:
                        mp.g=9.8
                    else:
                        mp.g=0
            elif event.type == pl.KEYUP:
                v = mp.DEFAULT_U

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
                    done, reason = env.game_over(env.history[frame][2])
                    if done:
                        print("reason", reason)
                    if mp.debug:
                        env.info(frame)
                last_frame = frame

        elif start and not done:
            done = exec_cmd(env, v, frame)
            frame = env.num_of_frames() - 1
            env.render(frame=frame)
            env.info()
            plot_data.append(env.history[frame])
            dump_plot(plot_data)
            if stepOne:
                start = False
                stepOne = False
            if done:
                print("done!")
                start = False
                replay = True


if __name__ == '__main__':
    main()

