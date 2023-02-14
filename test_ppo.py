from spinup.utils.test_policy import load_policy_and_env, run_policy
#from rl_pogo_phy import RabbitRLEnv
from rl_pogo_3point import RabbitRLEnv
import tensorflow as tf
import gym

_, get_action = load_policy_and_env('data/result-ppo')
env = RabbitRLEnv()
run_policy(env, get_action)
