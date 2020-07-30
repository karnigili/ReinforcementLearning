# Proximal Policy Optimization 

This repo demonstrates an implementation of a PPO using PyTorch. 

## A short intro to policy gradient methods



## About Proximal Policy Optimization 

tbd 

This is clipping

### Agent Componenets

### Neural Network Design

The agent contains two networks- a policy (actor) network and a value (critc) network, both follow the same network architecture, using the same hyperparameters.

#### Network Architecture  
The network is a straight forward sequential network with three hidden layers.
The input size is according to the state size, i.e., 33 nodes. Each hidden layer includes a rectified linear unit (ReLU) function and a linear transformation, using (configurable) 512 nodes. 
The output activation is a hyperbolic tangent function (tanh). Lastly, the output size follows the action size, 4 nodes.

### Model Hyperparameters

for more details, check out [this](https://medium.com/aureliantactics/ppo-hyperparameters-and-ranges-6fc2d29bccbe) great explanation.

* random seed`SEED` -  Randomization seed

A PPO agent collects trajectories according to the rolloutlength, 
then performs a gradient update of minibatch size on all the gathered 
trajectories for the specified number of epochs.

* Replay batch size `MINIBATCH_SIZE`- Number of experiences sampled in one batch
* Rollout length `ROLLOUT_LEN` - The size of the rollout
* Training epochs `N_EPOCHS` - The updating step iterations

* Discount factor `GAMMA` - Discount rate for future rewards
* Generalized Advantage Estimation  `GAE_LAMBDA`- Weights different bootstrap length estimation
* gradeint clipping `GRAD_CLIP` - Controls for exploding gradeints
* PPO clipping `PPO_CLIP` - The pre-update distance between policies
* Entropy Coefficient `ENTROPY_COEFF` - A regularizer (sets the max entropy for the policy)
* Value Function Coefficient `VF_COEFF` - 

* Learning rate `ALPHA` - Controls parameters update on the actor critic networks
* Epsilon `ADAM_EPS` - ADAM optimizer improve numerical stability 

* Episodes per training session `MAX_EPISODES`- Total number of episodes unless agent reached target performance earlier.
* Hidden layer size `HIDDEN_LAYER_N`- Number of hidden layer units

* Goal score `BENCHMARK` - The agent's mean goal score over 100 episodes

## Results

The 20 agents achieved the target score of 30 over 100 consecutive episodes. Upload the trained model using the [model.pth](https://github.com/karnigili/ReinforcementLearning/blob/master/Udacity-Deep-RL/PPO/PPO.pth) file.



## Future Directions 
