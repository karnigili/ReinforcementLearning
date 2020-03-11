### adapted from from https://towardsdatascience.com/reinforcement-learning-w-keras-openai-dqns-1eed3a5338c

import gym
import numpy as np
import random
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import Adam
import matplotlib.pyplot as plt

from collections import deque

from env_setting import env, goal_steps, goal_score, training_games

class DQN:
    def __init__(self, env):
        self.env     = env
        self.memory  = deque(maxlen=2000)

        self.gamma = 0.85
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.005
        self.tau = .125

        self.model        = self.create_model()
        self.target_model = self.create_model()

    def create_model(self):
        model   = Sequential()
        state_shape  = self.env.observation_space.shape
        model.add(Dense(24, input_dim=state_shape[0], activation="relu"))
        model.add(Dense(48, activation="relu"))
        model.add(Dense(24, activation="relu"))
        model.add(Dense(self.env.action_space.n))
        model.compile(loss="mean_squared_error",
            optimizer=Adam(lr=self.learning_rate))
        return model

    def act(self, state):
        self.epsilon *= self.epsilon_decay
        self.epsilon = max(self.epsilon_min, self.epsilon)
        if np.random.random() < self.epsilon:
            return self.env.action_space.sample()
        return np.argmax(self.model.predict(state)[0])

    def remember(self, state, action, reward, new_state, done):
        self.memory.append([state, action, reward, new_state, done])

    def replay(self):
        batch_size = 32
        if len(self.memory) < batch_size:
            return

        samples = random.sample(self.memory, batch_size)
        for sample in samples:
            state, action, reward, new_state, done = sample
            target = self.target_model.predict(state)
            if done:
                target[0][action] = reward
            else:
                Q_future = max(self.target_model.predict(new_state)[0])
                target[0][action] = reward + Q_future * self.gamma
            self.model.fit(state, target, epochs=1, verbose=0)

    def target_train(self):
        weights = self.model.get_weights()
        target_weights = self.target_model.get_weights()
        for i in range(len(target_weights)):
            target_weights[i] = weights[i] * self.tau + target_weights[i] * (1 - self.tau)
        self.target_model.set_weights(target_weights)

    def save_model(self, fn):
        self.model.save(fn)

def main():
    gamma   = 0.9
    epsilon = .95

    trials  = 1000
    trial_len = 500

    positions = np.ndarray([0,2])
    max_position=-.4
    rewards = []

    # updateTargetNetwork = 1000
    dqn_agent = DQN(env=env)
    steps = []
    for episode in range(trials):
        cur_state = env.reset().reshape(1,2)
        running_reward=0
        for step in range(trial_len):
            action = dqn_agent.act(cur_state)
            new_state, reward, done, _ = env.step(action)

            if new_state[0] > max_position:
                max_position = new_state[0]
                positions = np.append(positions, [[episode, max_position]], axis=0)
                running_reward += 10
            else:
                running_reward += reward
            # reward = reward if not done else -20
            new_state = new_state.reshape(1,2)
            dqn_agent.remember(cur_state, action, reward, new_state, done)

            dqn_agent.replay()       # internally iterates default (prediction) model
            dqn_agent.target_train() # iterates target model

            cur_state = new_state
            if done:
                rewards.append(running_reward)
                break
        if step >= 199:
            print("Failed to complete in trial {}".format(episode))
            if step % 10 == 0:
                dqn_agent.save_model("trial-{}.model".format(episode))
        else:
            print("Completed in {} trials".format(episode))
            dqn_agent.save_model("success.model")
            break

    plt.figure(1, figsize=[10,5])
    plt.subplot(211)
    plt.plot(positions[:,0], positions[:,1])
    plt.xlabel('Episode')
    plt.ylabel('Furthest Position')
    plt.subplot(212)
    plt.plot(rewards)
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.show()

if __name__ == "__main__":
    main()
