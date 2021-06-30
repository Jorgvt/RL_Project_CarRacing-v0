# RL_Project_CarRacing-v0
Repo where I'll be working in solving the CarRacing-v0 environment from gym.

# First approach

The first approach to the problem will be using `Stable Baselines` to get something done quick and easy. Once we get something working, we will start trying to implement the algorithm by hand.

An interesting point will be trying to use `Weights & Biases` to track our experiments of RL.

# Tips from Stable Baselines' Docs
- You should always do several runs to have quantitative results.
- Good results in RL are generally dependent on finding appropriate hyperparameters. Recent algorithms (PPO, SAC, TD3) normally require little hyperparameter tuning, however, don’t expect the default ones to work on any environment.
- Therefore, we highly recommend you to take a look at the RL zoo (or the original papers) for tuned hyperparameters. A best practice when you apply RL to a new problem is to do automatic hyperparameter optimization. Again, this is included in the RL zoo.
- When applying RL to a custom problem, you should always normalize the input to the agent (e.g. using VecNormalize for PPO/A2C) and look at common preprocessing done on other environments (e.g. for Atari, frame-stack, …).
- As a general advice, to obtain better performances, you should augment the budget of the agent (number of training timesteps).
- As some policy are stochastic by default (e.g. A2C or PPO), you should also try to set deterministic=True when calling the .predict() method, this frequently leads to better performance.
- Looking at the training curve (episode reward function of the timesteps) is a good proxy but underestimates the agent true performance.

## Choosing algorithm
There is no silver bullet in RL, depending on your needs and problem, you may choose one or the other. The first distinction comes from your action space, i.e., do you have discrete (e.g. LEFT, RIGHT, …) or continuous actions (ex: go to a certain speed)?
- The second difference that will help you choose is whether you can parallelize your training or not. If what matters is the wall clock training time, then you should lean towards A2C and its derivatives (PPO, …).

### Continous Actions - Single Process
Current State Of The Art (SOTA) algorithms are SAC, TD3 and TQC (available in our contrib repo). Please use the hyperparameters in the [RL zoo](https://github.com/DLR-RM/rl-baselines3-zoo) for best results.

### Continous Actions - Multiprocessed
Take a look at PPO or A2C. **Normalization is critical for those algorithms**.