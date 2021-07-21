import numpy as np
import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
import ptan
import wandb

class A2C(nn.Module):
    """
    Architecture based on Stable Baselines in order to make a fair comparison.
    """
    def __init__(self, obs_dim, action_dim):
        super(A2C, self).__init__()
        self.shared_core = nn.Sequential(*[
            nn.Linear(obs_dim, 128),
            nn.ReLU()
        ])
        self.policy = nn.Sequential(*[
            nn.Linear(obs_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim)
        ])
        self.value = nn.Sequential(*[
            nn.Linear(obs_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        ])
    
    def forward(self, X):
        # features = self.shared_core(X)
        ## In Stable Baselines, the shared core for vector observations is a flatten layer.
        # features = X.view(X.shape[0],-1)
        features = X
        action = self.policy(features)
        value = self.value(features)
        return action, value

def unpack_batch(batch, model, gamma, n_steps):
    states, actions, rewards, not_dones, last_states = [], [], [], [], []

    ## Unpack the elements in the batch
    for idx, exp in enumerate(batch):
        state = exp.state
        action = exp.action
        reward = exp.reward
        if exp.last_state is not None:
            not_done = idx
            last_state = exp.last_state
            not_dones.append(not_done)
            last_states.append(last_state)

        states.append(state)
        actions.append(action)
        rewards.append(reward)
        

    ## Now we have to add the last_state rewards.
    ## If its a terminal state, its value is 0.
    ## If its non terminal, its value can be estimated with the model.
    if not_done:
        last_states_t = torch.FloatTensor(last_states)
        last_states_values = model(last_states_t)[1].cpu().detach().numpy()
        rewards_np = np.array(rewards, dtype=np.float32) # Needs to be a np.array to allow indexing with a list
        rewards_np[not_dones] += (gamma ** n_steps)*last_states_values[:,0] # rewards_np is (80,) and last_state_values is (80,1)
    
    ## Turn everything into tensors
    states_t = torch.FloatTensor(states)
    actions_t = torch.LongTensor(actions)
    rewards_t = torch.FloatTensor(rewards_np)

    ## Return the tensors
    return states_t, actions_t, rewards_t

def see_model_in_action(env, model):
    """
    See how the model performs in the environment.
    The actions now are not sampled from the policy, we take the best one.
    
    Parameters
    ----------
    env: gym.Env
        The environment we want to train in.
    model: nn.Module
        Torch model.
    """
    state = env.reset()
    while True:
        env.render()
        state = torch.from_numpy(state).float()
        action, _ = model(state)
        ## When testing our environment we dont want to sample from the policy,
        ## we want to choose the best posible action.
        # action = Categorical(logits=action).sample()
        action = F.softmax(action, dim=-1).argmax()
        state, reward, done, info = env.step(action.item())
        if done:
            break
    env.close()


if __name__ == "__main__":
    
    SAVE_MODEL = True
    ## Define constants
    config = dict(
        ENV_ID = 'MountainCar-v0',
        REWARD_THRESHOLD = -110,
        # NORMALIZE_REWARD = True,
        N_STEPS = 5,
        N_ENVS = 16,
        GAMMA = 0.99,
        CLIP_GRAD = 0.5,
        BATCH_SIZE = 32, # Stable Baselines3 convention -> BATCH_SIZE = N_STEPS*N_ENVS 
        BETA_ENTROPY = 0.0,
        COEF_VALUE = 0.5,
        RMS_PROP_EPS = 1e-5,
        LEARNING_RATE = 7e-4,
        TIMESTEPS = 1e6
    )
    config["BATCH_SIZE"] = config["N_STEPS"]*config["N_ENVS"]

    run = wandb.init(config=config, project="RL_Test")
    config = wandb.config
    
    ## Set up the environment and the model
    ## We define it this way to allow multiple concurrent environments
    make_env = lambda: gym.make(config.ENV_ID)
    envs = [make_env() for _ in range(config.N_ENVS)]
    model = A2C(envs[0].observation_space.shape[0], envs[0].action_space.n)
    optimizer = torch.optim.RMSprop(model.parameters(), lr=config.LEARNING_RATE, eps=config.RMS_PROP_EPS)

    ## Tell WandB to track our model
    wandb.watch(model)

    ## Initialize agent and exp_source objects
    agent = ptan.agent.PolicyAgent(lambda x: model(x)[0], preprocessor=ptan.agent.float32_preprocessor, apply_softmax=True)
    exp_source = ptan.experience.ExperienceSourceFirstLast(envs, agent, gamma=config.GAMMA, steps_count=config.N_STEPS)

    ## Create empty list to store episode rewards
    ## and a completed episodes counter.
    rew = []
    completed_episodes = 0
    solved = False
    ## Create empty batch of experiences
    batch = []

    ## Fill the batch with experiences
    for steps, exp in enumerate(exp_source):
        batch.append(exp)
        
        ## pop_total_rewards() only returns something if an episode has been finished.
        ## In that case, it returns the accumulated reward of the episode.
        new_rewards = exp_source.pop_total_rewards()
        
        if new_rewards: # True when an episode has ended
            rew.append(new_rewards[0])
            completed_episodes += 1
            print(f"Steps: {steps} -> Reward: {np.mean(rew[-100:]):.2f}")
            
            if np.mean(rew[-100:]) > config.REWARD_THRESHOLD:
                solved = True
                print(f"Solved in {completed_episodes} episodes ({steps} steps)!")
                if SAVE_MODEL:
                    torch.save(model.state_dict(), f"{wandb.run.dir}/{config.ENV_ID}_{completed_episodes}_EP.pth")
                    torch.save(model.state_dict(), f"trained_models/{config.ENV_ID}_{completed_episodes}_EP.pth")
                    
                break
        
        ## If we still haven't filled the batch, restart the loop to get another experience
        if len(batch) < config.BATCH_SIZE:
            continue
        
        ## Break when we reach the desired number of timesteps
        if steps >= config.TIMESTEPS:
            print("Timed out!")
            break

        ## Unpack the experiences batch to get the training data
        states_t, actions_t, rewards_t = unpack_batch(batch, model, config.GAMMA, config.N_STEPS)
        batch.clear()
        ## Zero the gradients to start the training process
        optimizer.zero_grad()

        ## Get the policy logits and the values from the model
        policy_logits_t, values_t = model(states_t)

        ## Calculate the MSE loss between the values and the rewards
        loss_value_t = F.mse_loss(values_t.squeeze(-1), rewards_t)

        ## Calculate the advantage.
        ## Have to detach the values to avoid policy gradient propagating into values head.
        advantages_t = rewards_t - values_t.squeeze(-1).detach()

        ## Turn the logits into a probability distribution to facilitate the calculations.
        policy_dist = Categorical(logits=policy_logits_t)

        ## Calculate the policy log-lilkelihood with respect the actions taken.
        log_prob_t = policy_dist.log_prob(actions_t)
        
        ## Scale by the advantage
        policy_loss_t = -(advantages_t * log_prob_t).mean() # Minus sign because we want to maximize this function
        
        ## Calculate the entropy loss
        entropy_loss_t = policy_dist.entropy().mean()

        ## Add the losses
        loss_t = config.COEF_VALUE*loss_value_t + policy_loss_t - config.BETA_ENTROPY*entropy_loss_t

        ## Backpropagate the loss and update the weights. 
        ## Clip gradients if specified.
        loss_t.backward()
        if config.CLIP_GRAD:
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=config.CLIP_GRAD)
        optimizer.step()

        ## Log metrics
        wandb.log({
            "step":steps,
            "completed_episodes":completed_episodes,
            "value_loss":loss_value_t.item(),
            "policy_loss":policy_loss_t.item(),
            "entropy_loss":entropy_loss_t.item(),
            "total_loss":loss_t.item(),
            "advantage":advantages_t.mean().item(),
            "reward_mean_100":np.mean(rew[-100:])
        })

    ## Log last iteration metrics
    wandb.log({
        "step":steps,
        "completed_episodes":completed_episodes,
        "value_loss":loss_value_t.item(),
        "policy_loss":policy_loss_t.item(),
        "entropy_loss":entropy_loss_t.item(),
        "total_loss":loss_t.item(),
        "advantage":advantages_t.mean().item(),
        "reward_mean_100":np.mean(rew[-100:]),
        "solved":solved
    })
    wandb.run.summary["solved"] = solved
    ## Finish the logging
    wandb.finish()

    ## See the trained model in action!
    while True:
        env = gym.make(config.ENV_ID)
        see_model_in_action(env, model)