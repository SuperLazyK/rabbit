from spinup import ppo_pytorch as ppo
from rl_pogo_phy import RabbitRLEnv
import tensorflow as tf
import gym


env_fn = lambda : RabbitRLEnv()

ac_kwargs = dict(hidden_sizes=[1024, 512])

logger_kwargs = dict(output_dir='data/result-ppo', exp_name='rabbit')

# 150step/episode -> 5sec x 5 episode x 4cpu = 3000 step/epoch
ppo(env_fn=env_fn, ac_kwargs=ac_kwargs, steps_per_epoch=3000, epochs=3000, logger_kwargs=logger_kwargs)
