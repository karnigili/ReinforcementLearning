# Multi Agent Deep Deterministic Policy Gradient (DDPG) 

This repo demonstrates the implementation of a MADDPG using PyTorch. 

## A short intro to Deep Deterministic Policy Gradient
DDPG includes four networks; a local actor-critic and a target actor critic. Unlike classic actor critic, the DDPG actor directly maps states to actions rather than outputting a probability distribution across the action space.
The target networks are dealyed copies of the local networks which improves the stability in learning by preventing divergence in the update equations.

The pseudo code includes four main steps:
1. Experience replay 
    - using a replay buffer to store past experience
    - the replay ensures  the data to be independently distributed
2. Local actor & critic network updates
    - the critic, value, network is similar to Q-learning update. A big difference is that the next-state Q values are calculated with the **target** value network and target policy network.
    - the actor, policy, network maximizes the Q value via the mean of the sum of gradients calculated from the mini-batch.
3. Target network updates
    - these are updated via soft update, governed by the parameter tau. <img src="https://render.githubusercontent.com/render/math?math=Q' < \tau Q + (1-\tau) Q' ">
4. Exploration
   - in continuous action spaces, exploration is done via adding noise to the action itself. Here, using a Ornstein-Uhlenbeck process.



### Agent Componenets

The network includes a buffer and a noise generator. The buffer controls the batch learning by creating an internal memory for the agent, 
and the noise generator follows the Ornstein–Uhlenbeck process; it adds a noise parameter to improve learning (see [Continuous Control With Deep Reinforcement Learning](https://arxiv.org/abs/1509.02971) for more details)

### Neural Network Design

The agent contains two networks- a policy (actor) network and a value (critic) network, both follow a similar network architecture, using the same hyperparameters.

#### Network Architecture  
The network is a straight forward sequential network with three hidden layers. The input size is according to the state size, i.e., 24 nodes. Each hidden layer includes a rectified linear unit (ReLU) function and a linear transformation, using (configurable) 512 nodes for the first layer and 256 for the second one. 
The output activation is a hyperbolic tangent function (tanh). Lastly, the output size follows the action size, 4 nodes.

### Model Hyperparameters

* random seed`SEED` -  Randomization seed

* Replay batch size `MINIBATCH_SIZE`- Number of experiences sampled in one batch
* Memory size `BUFFER_SIZE` - the length of the internal memory

* Discount factor `GAMMA` - Discount rate for future rewards

* Learning rate 
    * `ALPHA` - Controls parameters update on the actor networks
    * `CRITIC_ALPHA` - Controls parameters update on the critic networks
* Interpolation parameter `TAU`  - soft update parameter

* Episodes per training session `MAX_EPISODES`- Total number of episodes unless agent reached target performance earlier.
* Hidden layer size `HIDDEN_LAYER_N`- Number of hidden layer units per layer

* Goal score `BENCHMARK` - The agent's mean goal score over 100 episodes

* Update rate `UPDATE_RATE` - policy update frequency

## Results

The 2 agents achieved the target score of 0.5 over 100 consecutive episodes. 
Upload the trained model using the [critic.pth]() and [actor.pth]() files.


## Future Directions 
1. Attempt a simple noise parameter, perhaps even a Gaussian ( see [here](https://arxiv.org/pdf/1802.09477.pdf) and [here](https://arxiv.org/pdf/1804.08617.pdf))
2. Try solving the problem with other multi-agent algorithms
