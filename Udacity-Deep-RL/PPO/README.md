
# Solving the Unity Reacher environment using PPO and PyTorch

![Reacher](reacher.gif)

This project features a policy-based RL algorithm. Here, a PPO agent solves the Unity Reacher environment.

## Proximal Policy Optimization 
Read more about PPO and this implementation at the [report.md]().

## The Environment
In this environment, a double-jointed arm can move to target locations. A **reward** of +0.1 is provided for each step that the agent's hand is in the goal location. Thus, **the goal** of your agent is to maintain its position at the target location for as many time steps as possible.

The observation space consists of 33 variables corresponding to position, rotation, velocity, and angular velocities of the arm. Each action is a vector with four numbers, corresponding to torque applicable to two joints. Every entry in the action vector should be a number between -1 and 1.

There are two version, 
* The first version contains a single agent.
* The second version contains 20 identical agents, each with its own copy of the environment. 

In order to solve the environment, 
* In the first version, the signle agent must get an average score of +30 over 100 consecutive episodes.
* In the second version, the average score of all 20 agents is +30 over 100 consecutive episodes. An average is recieved by adding up the rewards that each agent received (without discounting) after each episode, then take the average of these 20 scores.


## Getting started
### Dependencies
This project uses Unity mlagents & Pytorch. To install these, I used a conda environment. Follow the next steps to easily set up a local environment. 

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

**One Agent**

* Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher_Linux.zip)
* Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher.app.zip)
* Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher_Windows_x86.zip)
* Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher_Windows_x86_64.zip)

**Twenty (20) Agents**
* Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Linux.zip)
* Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher.app.zip)
* Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-* Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Windows_x86_64.zip)
    
When ready, launch a notebook using jupyter notebooks. 
~~~~
jupyter notebook
~~~~
### Running the agent
Open `Continuous-Control.ipynb` and run cell by cell. The code in this file loads the used libraries, builds a PPO agent, and, lastly, trains the model. 

______
Read more about the project [here](https://github.com/udacity/deep-reinforcement-learning/tree/master/p2_continuous-control)
