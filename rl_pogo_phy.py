import gym
from gym import spaces
from gym.utils import seeding
from env_point_pogo_phy import RabbitEnv
import numpy as np

class RabbitRLEnv(gym.Env):

    metadata = {
        'render.modes' : ['human', 'rgb_array'],
        'video.frames_per_second' : 30
    }

    def __init__(self):
        self.env = RabbitEnv()
        self.action_space = spaces.Box(low=self.env.min_action, high=self.env.max_action, dtype=np.float32)
        self.observation_space = spaces.Box(low=self.env.min_obs, high=self.env.max_obs, dtype=np.float32)
        self.seed()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def reset(self, random=True):
        return self.env.reset(self.np_random if random else None)

    def step(self, act):
        return self.env.step(act)

    def close(self):
        self.env.close()

