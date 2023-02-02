from spinup import ppo_pytorch as ppo
from env_point import RabbitEnv
import gym


env_fn = lambda : RabbitEnv()

ac_kwargs = dict(hidden_sizes=[64,64,64])

logger_kwargs = dict(output_dir='data/result-ppo', exp_name='rabbit')

ppo(env_fn=env_fn, ac_kwargs=ac_kwargs, steps_per_epoch=5000, epochs=2000, logger_kwargs=logger_kwargs)
