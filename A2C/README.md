# Actor-Critic Method

## Introduction

It is another extension to the vanilla policy gradient method, which improves the stability and convergence speed. It's one of the most powerful methods in deep reinforcement learning.

In REINFORCE we talked about reducing the variance by substracting a baseline from our scaling factors ($Q$-values). The next steo in reducing the variance is making our baseline state-dependent, which is a good idea since different states could have very different baselines. To decide on the suitability of a particular action in some state, we use the discounted total reward of the action. However, the total reward itself could be represented as a *value* of the state plus the *advantage* of the actions: $Q(s,a)=V(s)+A(s,a)$.

This means that if we use $V(s)$ as a baseline, the scale of our gradient will be just the *Advantage* $A(s,a)$, showing how this taken action is better in respect to the average state's value. **It's a very good idea for improving the policy gradient method.** The problem here is that we don't know the value of the state that we need to substract from the discounted total reward. To solve this, we can use another neural network which will approximate $V(s)$ for every observation.

When we know the value for any state we can use it to calculate the policy gradient and update our policy network to increase probabilities for actions with good advantage values and decrease the chance of actions with bad advantage values. The policy network is called the **actor** and the other network is called the **critic**, as it allows us to understand how good our actions were. This improvement is known as the **advantage actor-critic method**, **A2C**.

In practice, the policy and value networks partially overlap due to efficiency and convergence considerations. They are implemented as different heads of the network. This helps both networks to share low-level features but combine them in a different way.

## Training

1. Initialize network parameters $\theta$ with random values.
2. Play $N$ steps in the environment using the current policy $\pi_{\theta}$ and saving the state $s_{t}$, the action $a_{t}$ and the reward $r_{t}$.
3. $R=0$ if the end of the episode is reached or $V_{\theta}(s_{t})$.
4. For $i=t-1...t_{start}$ (the steps are processed backward):
    - $R \larr r_{i} + \gamma R$
    - Accumulate the policy gradients: $\partial \theta_{\pi} \larr \partial \theta_{\pi} + \nabla_{theta} log\pi_{\theta}(a_{i}|s_{i})\left( R-V_{\theta}(s_{i}) \right)$
    - Accumulate the value gradients: $\partial \theta_{v} \larr \partial \theta_{v} + \frac{\left( R-V_{\theta}(s_{i}) \right)^{2}}{\partial \theta_{v}}$
5. Update the network parameters using the accumulated gradients. Moving in the direction of the policy gradients and in the opposite direction of the value gradients.
6. Repeat from step 2 until convergence is reached.

In practice, some considerations are added as follows:
- An entropy bonus is usually added to improve exploration.
- Gradient accumulation is usually implemented as a loss function combining all three components: policy loss, value loss and entropy loss.
- It's worth using several environments to improve stability. This version of the algorithm is called **advantage asynchronous actor-critic**, also known as **A3C**.