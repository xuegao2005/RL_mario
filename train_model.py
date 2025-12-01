import gym
from numpy import shape
from stable_baselines3 import PPO
from nes_py.wrappers import JoypadSpace
import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT

env = gym_super_mario_bros.make('SuperMarioBros-v2')
env = JoypadSpace(env, SIMPLE_MOVEMENT)

model = PPO("CnnPolicy", env, verbose=1, tensorboard_log="logs")
model.learn(total_timesteps=1000)

model.save("ppo_mario")
