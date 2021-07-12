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

## Key takeaways from the implementation

- In order to generate a training set, we make the policy play and store the pairs (state, action, Q-value). Q-value is calculated on the fly from the rewards.
- We then make a forward pass with the states through our policy network to generate its corresponding probability distribution over the actions.
- Take same action that we took when generating the training set (that's why we store which action was performed). This is specially important because the actions weren't chosen with an `argmax` but sampling from the output distribution, so we need to make sure we're taking the same action as before.
- Now we can calculate the loss!

## REINFORCE issues

- Full episodes are required. Both REINFORCE and the cross-entropy method behave better with more episodes used ofr training. This situation is fine for short episodes in Cartpole, but in Pong every episode can last for hundreds or even thousands of frames. The purpose of the complete episode requirement is to get as accurate a Q-estimation as possible. In DQN it's fine to replace the exact value for a discounted reward using the Bellman equation, but in the case of the policy gradient, we don't have $V(s)$ or $Q(s,a)$ anymore.
To overcome this, there are two approaches:
    - We can ask our network to estimate $V(s)$ and use this estimation to obtain $Q$. This approach is called the **actor-critic method**.
    - We can do the Bellman equation, unrolling $N$ steps ahead, which will effectively exploit the fact that the value contribution decreases when gamma is less than 1.

- High gradients variance. In the policy gradient formula we have a gradient proportional to the discounted reward from the given state. However, the range of this reward is heavily environment-dependent. For example, in the CartPole environment, we get a reward of 1 for every timestamp that we are holding the pole. Which means that unsuccessful samples will be quite lower than successful ones, and such a large difference can seriously affect our training dynamics, as one lucky episode will dominate the final gradient.
In mathematical terms, the policy gradient has high variance, and we need to do something about this or the training process can become unstable. The usual approach to handling this is substracting a value called the **baseline** from the $Q$. Posible choices for baselines:
    - Constant value. Normally the mean of the discounted reward.
    - Moving average of the discounted rewards.
    - Value of the state, $V(s)$.

- **Exploration**. There is high chance that the agent will converge to some locally optimal policy and stop exploring the environment. Policy gradient methods allow us to use the **entropy bonus**. Entropy is a measure of uncertainty and can show how much the agent is uncertain about which action to take. To prevent our agent from being stuck in the local minimum, we substract the entropy from the loss function, punishing the agent for being too certain about the action to take.

- **Correlation between samples**. Training samples in a single episode are usually heavily correlated, which is bad for SGD. TO solve this, parallel environments are normally used. Instead of communicating with one environment, we use several and use their transitions as training data.