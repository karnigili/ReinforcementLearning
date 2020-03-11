# %% codecell
from keras.optimizers import Adam
from keras.models import Sequential, Model
from keras.layers import Dense, Input, Reshape, Lambda
from keras.layers.merge import concatenate
from keras import backend as K
import matplotlib.pyplot as plt

from env_setting import env, goal_steps, goal_score, training_games,\
    state_space_n, action_space_n

import random
import numpy as np

### adapted from  https://blog.tanka.la/2018/10/19/solving-curious-case-of-mountaincar-reward-problem-using-openai-gym-keras-tensorflow-in-python/
class Behvaioral_Cloning(object):
    """
    Behvaioral Cloning keras agent
    """
    def __init__(self, env=env, training_games=training_games, goal_score=goal_score, goal_steps=goal_steps):
        self.env=env

        ## data prep
        self.training_games=training_games
        self.goal_score=goal_score
        self.goal_steps=goal_steps
        self.position_threshold=-0.2


    def data_preparation(self):
        ## make useful data for training
        # this function only chooses episodes that were sucsseful
        # based on a threshold set in advance

        training_data=[]
        accepted_scores=[]

        for game_index in range(self.training_games):
            score=0

            track_game_results=[]
            # observation >> the current position of Car and its velocity
            prev_state=[]
            to_track=False

            for step_index in range(self.goal_steps):
                # choose a random action
                # push left(0), no push(1) and push right(2)
                action=self.env.action_space.sample()
                #keep track on step properties (progress)
                observation, reward, done, info=self.env.step(action)

                #add previous data to memory (from second step onwards)
                if to_track:
                    track_game_results.append([prev_state, action])

                # move observation
                prev_state=observation
                to_track=True

                #define success threshold position
                if observation[0]>self.position_threshold:
                    reward=1

                score+=reward
                if done:
                    # done if reached 0.5(top) position
                    break

                # otherwise, runs until 200 iterations are reached

            # if reached goal score
            if score>=self.goal_score:
                #consider a success
                accepted_scores.append(score)
                for data in track_game_results:
                    kinematics=data[0]
                    action=data[1]
                    # hot encode action info
                    if action==1:
                        output = [0, 1, 0]
                    elif action==0:
                        output = [1, 0, 0]
                    elif action==2:
                        output = [0, 0, 1]
                    # record kinematics and categorical action info
                    training_data.append([kinematics, output])

            env.reset()

        self.training_data=training_data

    ## BUILD ASYNC MODEL ##
    def create_model(self, input_size, output_size):
        ## build a model with keras

        state=Input((input_size,))

        layer=Dense(24, input_dim=input_size, activation="relu")(state)
        layer=Dense(48, activation="relu")(layer)
        layer=Dense(24, activation="relu")(layer)

        value=Dense(output_size, activation='linear', name='value')(layer)
        value_network=Model(input=state, output=value)

        return value_network
    def train_model(self):
        '''
        training_data includes pairs of states and actions
        states are of shape (2,)
        actions are hot encoded to shape (3,)
        '''
        # X is the obs (kinematics) and y is the output
        X=np.array([i[0] for i in self.training_data]).reshape(-1, len(self.training_data[0][0]))
        y=np.array([i[1] for i in self.training_data]).reshape(-1, len(self.training_data[0][1]))

        self.model=self.create_model(input_size=len(X[0]), output_size=len(y[0]))

        self.model.compile(loss='mse', optimizer=Adam())
        self.model.fit(X, y, epochs=5)

    #play 100 games with the trained model
    def play_using_trained_model(self, render=False):

        ## let's see the results
        # track results
        scores=[]
        choices=[]

        positions = np.ndarray([0,state_space_n])
        max_position=-.4
        rewards = []

        for episode in range(1000):
            score=0
            prev_state=env.reset()#.reshape(1,2)
            running_reward=0

            for step_index in range(500):#self.goal_steps):

                #only render last 10 games
                if render and episode>90:
                     self.env.render()

                #if len(prev_state)==0:
                #action=random.randrange(0,action_space_n-1)
                #else:
                action=np.argmax(self.model.predict(prev_state.reshape(-1, \
                    len(prev_state)))[0])

                choices.append(action)
                new_observation, reward, done, info = self.env.step(action)
                prev_state=new_observation

                score+=reward

                if new_observation[0] > max_position:
                    max_position = new_observation[0]
                    positions = np.append(positions, [[episode, max_position]], axis=0)

                    running_reward += 10
                else:
                    running_reward += reward

                if done:
                    rewards.append(running_reward)
                    break

            self.env.reset()
            scores.append(score)

        print('Average Score:',sum(scores)/len(scores))
        print('choice 1:{}  choice 0:{} choice 2:{}'.format(choices.count(1)/len(choices),choices.count(0)/len(choices),choices.count(2)/len(choices)))
        #return [range(100),np.mean(tot_scores,axis=1)]

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
    #get training data
    #training_data=model_data_preparation(env, training_games, goal_steps, goal_score)
    #build a model
    #trained_model=train_model(training_data)
    #let it play the game
    #play_using_trained_model(trained_model, training_data, goal_steps)
    AGENT=Behvaioral_Cloning()
    AGENT.data_preparation()
    AGENT.train_model()
    AGENT.play_using_trained_model(render=False)
