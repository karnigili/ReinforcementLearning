# ICM for the mountain car problem


This repo includes code presenting the mountain car problem (taken from OpenAI gym environment). Further, it demonstrates different RL algorithms and their performance at solving this problem.

## Files
The python files in this directory are as follows
1. *env_setting.py* imports the environment and manages the overarching configuration parameters
2. *car_demo.py* a simple implementation of a random run of the environment
3. *dqn.py* code adapted from [here](### adapted from from https://towardsdatascience.com/reinforcement-learning-w-keras-openai-dqns-1eed3a5338c) implementing deep Q-learning on the mountain car problem
4. *supervised_cloning.py* code implementing a supervised learning approach to solve the mountain car problem
5. *icm.py* an ICM model, inspired by [this repo](https://github.com/neka-nat/async-rl-noreward/tree/3da83655745a82f3184daac746439228f40410b6)

This directory also includes a few figures showing the agents' performance.  
1. *dqn_mc* shows the deep Q-learning agent performance
2. *icm_mc* shows the ICM agent performance

## Requirements
- keras
- gym

### Pay attention, this is a DRAFT. This project is still unfinished.
