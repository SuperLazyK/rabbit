
#class RabbitLearningEnv(gym.Env):
#
#    metadata = {
#        'render.modes' : ['human', 'rgb_array'],
#        'video.frames_per_second' : FRAME_RATE
#    }
#
#    def __init__
#        #max_action = np.array([MAX_TORQUE1, MAX_TORQUE2])
#        max_obs    = np.array([ np.pi/2 , np.pi/2 , np.pi/2 , MAX_ANGV , MAX_ANGV , MAX_ANGV ], dtype=np.float32)
#
#        self.action_space = spaces.Discrete(4)
#        #self.action_space = spaces.Box(low=-max_action, high=max_action, dtype=np.float32)
#        self.observation_space = spaces.Box(low=-max_obs, high=max_obs, dtype=np.float32)
#        self.seed()
#
#    def seed(self, seed=None):
#        self.np_random, seed = seeding.np_random(seed)
#        return [seed]
#
#    def step(self, act):
#        return s, reward, done, {}

