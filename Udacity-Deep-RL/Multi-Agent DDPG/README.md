
# Solving the Unity Tennis environment using MA DDPG and PyTorch

![img](https://github.com/karnigili/ReinforcementLearning/blob/master/Udacity-Deep-RL/Multi-Agent%20DDPG/tennis.gif)

This project features a policy-based RL algorithm. Here, two DDPG agent play tennis in a Unity environment.

## Deep Deterministic Policy Gradients 
Read more about PPO and this implementation at the 
[report.md](https://github.com/karnigili/ReinforcementLearning/blob/master/Udacity-Deep-RL/Multi-Agent DDPG/report.md).

## The Environment
In this environment, two agents control rackets to bounce a ball over a net. If an agent hits the ball over the net, it receives a reward of +0.1.  If an agent lets a ball hit the ground or hits the ball out of bounds, it receives a reward of -0.01.  
Thus, the goal of each agent is to keep the ball in play. The observation space consists of 33 variables corresponding to position, rotation, velocity, and angular velocities of the arm. Each action is a vector with four numbers, corresponding to torque applicable to two joints. Every entry in the action vector should be a number between -1 and 1.

The observation space consists of 8 variables corresponding to the position and velocity of the ball and racket. 
Each agent receives its own, local observation.  Two continuous actions are available, corresponding to movement toward (or away from) the net, and jumping. 

The task is episodic, and in order to solve the environment:
The agents must get an average score of +0.5 (over 100 consecutive episodes, after taking the maximum over both agents). 
Specifically,

- After each episode, we add up the rewards that each agent received (without discounting), to get a score for each agent. This yields 2 (potentially different) scores. We then take the maximum of these 2 scores.
- This yields a single **score** for each episode.

The environment is considered solved, when the average (over 100 episodes) of those **scores** is at least +0.5.


## Getting started
### Dependencies
This project uses Unity ML-agents & Pytorch. To install these, I used a conda environment. Follow the next steps to easily set up a local environment. 

First, install [Anaconda](https://www.anaconda.com/download/). Then, run in in terminal (Notice, this code is optimized for MacOS):
~~~~
conda install python=3.6
conda create -n py36 python=3.6
source activate py36 (Do this everytime you open the terminal)
pip install "tensorflow==1.7.0"
conda install -c anaconda jupyter
pip install mlagents 
pip install unityagents
pip install torch
~~~~

Next, download the environment. Download the relevant file, place it in your working directory, and unzip it. Notice to modify the name of the environment in the main file to match the name of the decompressed file.

environment that matches your operating system:
* Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Linux.zip)
* Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis.app.zip)
* Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Windows_x86.zip)
* Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Windows_x86_64.zip)

When ready, launch a notebook using jupyter notebooks. 
~~~~
jupyter notebook
~~~~
### Running the agent
Open `DDPG.ipynb` and run cell by cell. The code in this file loads the used libraries, builds a DDPG agent, and, lastly, trains the model. 

______
Read more about the project [here](https://github.com/udacity/deep-reinforcement-learning/tree/master/p3_collab-compet)
