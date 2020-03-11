from get_car_env import env, goal_steps, goal_score, training_games

import random
import numpy as np

import matplotlib.pyplot as plt

print('import done')


states = env.observation_space.shape[0]
actions = env.action_space.n

print('This environment has {} actions and {} states'.format(actions,states))



def play_one_random_game():
    for step_index in range(goal_steps):
        # visualize each step
        env.render()
        # choose a random action
        action=env.action_space.sample()
        #keep track on step properties (progress)
        observation, reward, done, info=env.step(action)

        #print
        print("Step number {}:".format(step_index))
        print("action taken: {}".format(action))
        print("observation: {}".format(observation))
        print("reward: {}".format(reward))
        print("done: {}".format(done))
        print("info: {}".format(info))
        if done:
            break
    # At each begining reset the game
    env.reset()

play_one_random_game()
