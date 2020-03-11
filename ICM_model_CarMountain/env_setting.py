import gym
from gym import envs
import matplotlib.pyplot as plt


env = gym.make('MountainCar-v0')

env.reset()
goal_steps=200
goal_score=-198
training_games=10000

state_space_n=2
action_space_n=3
