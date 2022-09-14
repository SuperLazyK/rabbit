from spinup.utils.test_policy import load_policy_and_env, run_policy
from env_point import RabbitEnv
import tensorflow as tf
import gym

_, get_action = load_policy_and_env('data/result-ppo')
env = RabbitEnv()
run_policy(env, get_action)
