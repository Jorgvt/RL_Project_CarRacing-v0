# REINFORCE

Our first step towards getting our own implementation of **PPO** is going to be implementing **REINFORCE**. We do so because it's the first and easiest **policy gradient** algorithm. The idea is that once we have it implemented, we can add variations to it in order to achieve more complicated algorithms as **A2C** and **PPO**.

## Introduction to REINFORCE

The **policy gradient** is defined as: $\Delta J \approx \mathbb{E}\left[ Q(s,a) \nabla log\pi(a|s) \right]$. Defines the direction in which we need to change our network's parameters to improve the policy in terms of teh accumulated total reward. The scale of the gradient is proportional to the value of the action taken, which is $Q(s,a)$. One very important point is how exactly gradient scales (Q(s,a)) are calculated. The idea is to use Q(s,a) to increase probabilities of good actions in the beginning of the episode and decrease the actions closer to the end of the episode. As Q(s,a) incorporates the discount factor, uncertainty for longer sequences of actions is automatically taken into account. This is the idea behind **REINFORCE**.

## Algorithm

1. Initialize the network with random weights.
2. Play $N$ full episodes, saving their $(s, a, r, s')$ transitions.
3. For every step, $t$, of every episode, $k$, calculate the discounted total reward for subsequent steps: $Q_{k,t} = \sum_{i=0}\gamma^{i}r_{i}$.
4. Calculate the loss function for all transitions: $\mathcal{L} = -\sum_{k,t}Q_{k,t}log\left( \pi\left( s_{k,t},a_{k,t} \right) \right)$
5. Perform and SGD update of weights, minimizing the loss.
6. Repeat from step 2 until converged.

## Differences with Q-learning

- **No explicit exploration is needed**. As the network returns probabilities, the exploration is performed automatically (because we choose our actions sampling from that probability distribution). In the beginning, the network is initialized withj random weights, and it returns a uniform probability distribution. This distribution corresponds to a random agent behaviour.
- **No replay buffer is used**. Policy gradient methods are **on-policy**, which means that we can't train on data obntained from the old policy. Such methods usually converge faster, but require much more interaction with the environment than off-policy methods.
- **No target network is needed**. We use Q-values, but they are obtained from our experience in the environment. In DQN we used the target network to break the correlation in Q-values approximation, but we are not approximating anymore. The target network trick can still be useful in policy gradient methods.