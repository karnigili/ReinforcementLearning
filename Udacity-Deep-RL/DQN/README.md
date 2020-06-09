# Solving the Unity Banana-Navigation task using DQN and PyTorch

![banana](banana.gif)

This project features a value-based RL algorithm. Here, a DQN agent solves the Unity Banana-Navigation task.

## Deep Q-learning 
Read more about Deep Q-learning and this implementation at the [report.md](https://github.com/karnigili/ReinforcementLearning/blob/master/Udacity-Deep-RL/DQN/report.md) file.

## The Environment
The environment describes a 2D space filled with yellow and blue bananas. The agent moves around the space collecting bananas. 

A **reward** of +1 is provided for collecting a yellow banana, and a reward of -1 is provided for collecting a blue banana. Thus, **the goal** of your agent is to collect as many yellow bananas as possible while avoiding blue bananas.

The state space has 37 dimensions and contains the agent's velocity, along with ray-based perception of objects around agent's forward direction. Given this information, the agent has to learn how to best select actions. Four discrete actions are available, corresponding to:

* `0` - move forward.
* `1` - move backward.
* `2` - turn left.
* `3` - turn right.

The task is episodic, and in order to solve the environment, your agent must get an average score of +13 over 100 consecutive episodes.


## Getting started
### Dependencies
This project uses Unity mlagents & Pytorch. To install these, I used a conda environment. Follow the next steps to easily set up a local environment. 

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

Next, download the environment (here, I use a variation of the Unity Banana task, created by Udacity). Download the relevant file, place it in your working directory, and unzip it. Notice to modify the name of the environment in the main file to match the name of the decompressed file.
* Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Linux.zip)
* Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana.app.zip)
* Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Windows_x86.zip)
* Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Windows_x86_64.zip)
    
When ready, launch a notebook using jupyter notebooks. 
~~~~
jupyter notebook
~~~~
### Running the agent
Open `Navigation.ipynb` and run cell by cell. The code in this file loads the used libraries, builds a DQN agent, and, lastly, trains the model. 

______
Read more about the project [here](https://github.com/udacity/deep-reinforcement-learning/tree/master/p1_navigation)
