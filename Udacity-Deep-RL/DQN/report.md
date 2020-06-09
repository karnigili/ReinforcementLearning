# Deep Q Learning (DQN)

This repo demonstrates an implementation of a vanilla DQN using PyTorch. 

## Intro to Q Learning

Q Learning is a relatively intuitive RL algorithm that maximizes the total reward by finding the best action to take given the current state. Q Learning is an off-policy, value-based approach.
It learns to match action to the state by performing out-of current policy actions. Simply, the Q function is updated given


<img src="https://render.githubusercontent.com/render/math?math=Q(s_t,a_t)\=(1-\alpha)Q(s_t,a_t) %2B \alpha(r_t %2B \gamma \max_a Q(s_{t%2B1},a))">


The Q function dictates the policy governing the agent's behavior, given a state. The updating process includes selecting the action that results in the maximum value for the given state at each step. `alpha` describes the earning rate and `gamma` the discount factor.

To take actions, a Q-learning commonly follows an action selection policy, such as `epsilon` greedy (somewhat detached from the policy given by the Q-table, i.e, off-policy). The agent learns by observing the environment at each step, and updating the Q-table given the state, action, and reward following the above equation.  

(*) Importantly, Q-learning only works in environments with discrete and finite state and action spaces. However, using function approximators can help overcome this issue.


## Deep Q Learning

The goal is to identify the policy that best maximizes the Q function. However, given that we do not obtain all data in advance, we can not merely construct such a policy. We utilize neural networks being universal function approximators to resemble such a policy.

Following DeepMind's [paper](https://web.stanford.edu/class/psych209/Readings/MnihEtAlHassibis15NatureControlDeepRL.pdf), I used a replay with a separate target network. *Replay* (also known as experience replay or memory replay) describes storing and reusing the transitions, i.e. the agent's experience.   
The separate target network enables the replay by calculating the value of the state reached as a result of action independently from the policy network.

This implementation overcomes an issue imposed by deep learning, diverging. It supports stabilizing and improves the training process of Q-Learning. First, a random sampling of the experiences from the memory decorates the data points used to update the network. Second, it increases the learning speed via the use of mini-batches. See the `ReplayMemory` class.

The learning happens as follows (see the figure below for general guidance or `train` function).
1. Actions are chosen based on the greedy `epsilon` approach (thus, either randomly or based on a policy (local) network).
2. The action yields a new environment observation. The current state, action, and reward from this observation are 
    * recorded in the replay memory and 
    * used to update the target network (given the update rate)
3. Given the update rate, the policy (local) network is updated using a random sample from memory. 


![img](https://pytorch.org/tutorials/_images/reinforcement_learning_diagram.jpg)

### Neural Network Design

The target network and the policy (local) network both follow the same network architecture, using the same hyperparameters.

#### Network Architecture  
The network is a straight forward sequential network with two hidden layers.   The input size is according to the state size, i.e., 37 nodes. Each hidden layer includes a  rectified linear unit (ReLU) function and a linear transformation, using 64 nodes. Lastly, the output size follows the action size, 4 nodes.

#### Network Hyperparameters

* random seed`SEED` -  Randomization seed
* buffer size `BUFFER_SIZE` - Maximum size of experience replay memory
* Replay batch size `BATCH_SIZE`- Number of experiences sampled in one batch

* Discount factor `GAMMA` - Discount rate for future rewards
* Soft update factor `TAU` Controls parameters update of the target Q-network from the online Q-network 
* Learning rate `ALPHA` - Controls parameters update of the online Q-network 
* Target update frequency `UPDATE_RATE`- Number of learning steps between parameter updates of the target Q-network

* Episodes per training session `MAX_EPISODES`- Total number of episodes unless agent reached target performance earlier.
* Time steps per episode `TIME_STEPS` - Length of each episode in time steps.
* Epsilon start `INIT_EPSILON`- Start value for the epsilon parameter in epsilon-greedy strategy
* Epsilon end
 `MIN_EPSILON` - Final value for the epsilon parameter in epsilon-greedy strategy
* Epsilon decay-`EPSILON_DECAY`- Decrease of the epsilon parameter in epsilon-greedy strategy


## Results

The agent achieved the target score of 13 over 100 consecutive episodes, as seen in the graph below. Upload the trained model using the [model.pth](https://github.com/karnigili/ReinforcementLearning/blob/master/Udacity-Deep-RL/DQN/model.pth) file.


![img](https://github.com/karnigili/ReinforcementLearning/blob/master/Udacity-Deep-RL/DQN/results.png)


## Future Directions 

A straight forward idea to continue this project would be to implement a more complex DQN version. There are several issues with the DQN algorithm and at least six common extensions that address these different issues. The figure below compares their performance. 

![img](https://video.udacity-data.com/topher/2018/June/5b3814f1_screen-shot-2018-06-30-at-6.40.09-pm/screen-shot-2018-06-30-at-6.40.09-pm.png)

A good place to start would be to choose a simple extension, like the Double DQN (DDQN), the Prioritized experience replay, or the Dueling DQN. 

**Double DQN** resolves a stability issue with DQN. Specifically, it targets the maximization bias stemming from a systematic overestimation in the learning process. DDQN introduces two separate Q-value estimators, each of which is used to update the other. This separation decreases the bias and stabilizes the learning process. Read more about it [here](https://arxiv.org/pdf/1509.06461.pdf).

**Prioritized experience replay** improves the sampling of previous experience. In the classic DQN, experience transitions were uniformly sampled from a replay memory. However, that ignores the idea that some experiences hold higher significance compared to others. Such significance is commonly attributed using the TD error. Read more about it [here](https://arxiv.org/abs/1511.05952).

**Dueling DQN** enhances the classic DQN by generalizing the learning across actions without imposing any change to the underlying reinforcement learning algorithm. It does so by creating two separate estimators: one for the state value function and the other for state-dependent action advantage function. Read more about it [here](https://arxiv.org/abs/1511.06581).
