# Proximal Policy Optimization 

This repo demonstrates the implementation of a PPO using PyTorch. 

## A short intro to policy gradient methods

Shortly, policy gradient methods optimize the parametrized policies concerning the expected return (long-term cumulative reward), commonly, using gradient descent. See the equation below.

<img src="https://render.githubusercontent.com/render/math?math=\hat{g}=\hat{E_t}[\nabla_\theta \log{\pi_{\theta}(\alpha_t|s_t)}\hat{A}_t]">

Where <img src="https://render.githubusercontent.com/render/math?math=\pi_{\theta}"> is a stochastic policy and <img src="https://render.githubusercontent.com/render/math?math=\hat{A}_t"> is the advantage function estimation at time *t*. The expected value <img src="https://render.githubusercontent.com/render/math?math=E [...]"> indicates batch training.

Policy gradients iterate between estimating the advantage for the current policy and using the approximated advantage function to improve the next policy. However, this can raise issues when the policy updates grow too large. How does PPO tackle this?

## About Proximal Policy Optimization 

PPO formalizes a constraint on the update gap as a penalty in the objective function. It implements a trust region, a maximum step size which limits the exploration. Classically, the trust region is set according to a KL-divergence measuring the distance between the current policy and the previous one. KL-divergence is computationally heavy, so it is popular to use a clipped Objective. In this method, we measure the ratio between the new policy with samples from the old policy (following the logic of importance sampling). We synchronize the older network with the newer one every few steps. Thus, the new objective function is - 

<img src="https://render.githubusercontent.com/render/math?math=L^{CLIP}_{\theta_k}(\theta) = E_{\tau~\pi_k} [\sum^T_t=0 [min(r_t(\theta)\hat{A}_t , clip(r_t(\theta), 1\-\epsilon, 1 \+ \epsilon)\hat{A}_t]]">

If the ratio between the policies (*r*) exceeds the range <img src="https://render.githubusercontent.com/render/math?math=1-\epsilon , 1+\epsilon"> , the advantage function will be clipped. This clipping discourages large policy change.

### Agent Componenets

The network includes a buffer and a rollout. The buffer controls the batch learning, and the rollout governs the sampling process for the policy update.

### Neural Network Design

The agent contains two networks- a policy (actor) network and a value (critic) network, both follow the same network architecture, using the same hyperparameters.

#### Network Architecture  
The network is a straight forward sequential network with three hidden layers. The input size is according to the state size, i.e., 33 nodes. Each hidden layer includes a rectified linear unit (ReLU) function and a linear transformation, using (configurable) 512 nodes. 
The output activation is a hyperbolic tangent function (tanh). Lastly, the output size follows the action size, 4 nodes.

### Model Hyperparameters

For more details, check out [this](https://medium.com/aureliantactics/ppo-hyperparameters-and-ranges-6fc2d29bccbe) great explanation.

* random seed`SEED` -  Randomization seed

A PPO agent collects trajectories according to the rollout length, 
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
* Value Function Coefficient `VF_COEFF` - A loss coefficient

* Learning rate `ALPHA` - Controls parameters update on the actor-critic networks
* Epsilon `ADAM_EPS` - ADAM optimizer improve numerical stability 

* Episodes per training session `MAX_EPISODES`- Total number of episodes unless agent reached target performance earlier.
* Hidden layer size `HIDDEN_LAYER_N`- Number of hidden layer units

* Goal score `BENCHMARK` - The agent's mean goal score over 100 episodes

## Results

The 20 agents achieved the target score of 30 over 100 consecutive episodes. Upload the trained model using the [model.pth](https://github.com/karnigili/ReinforcementLearning/blob/master/Udacity-Deep-RL/PPO/PPO.pth) file.


## Future Directions 
This agent sets the value of the hyperparameter according to the [PPO paper](https://arxiv.org/pdf/1707.06347.pdf). A possible extension would be the explore hyperparameter tuning and try to optimize the network. Additionally, I would like to try and run PPO on other environments and evaluate its performance on a broader scope.
