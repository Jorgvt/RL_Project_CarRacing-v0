import numpy as np
import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
import ptan

class A2C(nn.Module):
    def __init__(self, obs_dim, action_dim):
        super(A2C, self).__init__()
        self.shared_core = nn.Sequential(*[
            nn.Linear(obs_dim, 128),
            nn.ReLU()
        ])
        self.policy = nn.Sequential(*[
            nn.Linear(128, action_dim)
        ])
        self.value = nn.Sequential(*[
            nn.Linear(128, 1)
        ])
    
    def forward(self, X):
        features = self.shared_core(X)
        action = self.policy(features)
        value = self.value(features)
        return action, value

def unpack_batch(batch, model, gamma, n_steps):
    states, actions, rewards, dones, last_states = [], [], [], [], []

    ## Unpack the elements in the batch
    for exp in batch:
        state = exp.state
        action = exp.action
        reward = exp.reward
        done = exp.last_state is None
        last_state = exp.last_state

        states.append(state)
        actions.append(action)
        rewards.append(reward)
        dones.append(done)
        last_states.append(last_state)

    ## Now we have to add the last_state rewards.
    ## If its a terminal state, its value is 0.
    ## If its non terminal, its value can be estimated with the model.
    ## This is very innefficient, but its good enough for a simple test.
    for idx, last_state in enumerate(last_states):
        if last_state is not None:
            last_state = torch.FloatTensor(last_state).unsqueeze(0)
            value = model(last_state)[1].cpu().detach().numpy().item()
            rewards[idx] += (gamma**n_steps) * value
    
    ## Turn everything into tensors
    states_t = torch.FloatTensor(states)
    actions_t = torch.LongTensor(actions)
    rewards_t = torch.FloatTensor(rewards)

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
    ## Define constants
    ENV_ID = 'CartPole-v1'
    # NORMALIZE_REWARD = True
    N_STEPS = 5
    N_ENVS = 16
    GAMMA = 0.99
    BATCH_SIZE = 32 # Stable Baselines3 convention -> BATCH_SIZE = N_STEPS*N_ENVS 
    BETA_ENTROPY = 0.0
    COEF_VALUE = 0.5

    RMS_PROP_EPS = 1e-5
    LEARNING_RATE = 7e-4
    TIMESTEPS = 1e6
    SAVE_MODEL = True

    ## Set up the environment and the model
    ## We define it this way to allow multiple concurrent environments
    make_env = lambda: gym.make(ENV_ID)
    envs = [make_env() for _ in range(N_ENVS)]
    model = A2C(envs[0].observation_space.shape[0], envs[0].action_space.n)
    optimizer = torch.optim.RMSprop(model.parameters(), lr=LEARNING_RATE, eps=RMS_PROP_EPS)

    ## Initialize agent and exp_source objects
    agent = ptan.agent.PolicyAgent(lambda x: model(x)[0], preprocessor=ptan.agent.float32_preprocessor, apply_softmax=True)
    exp_source = ptan.experience.ExperienceSourceFirstLast(envs, agent, gamma=GAMMA, steps_count=N_STEPS)

    ## Create empty list to store episode rewards
    ## and a completed episodes counter.
    rew = []
    completed_episodes = 0
        
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
            
            if np.mean(rew[-100:]) > 475:
                print(f"Solved in {completed_episodes} episodes ({steps} steps)!")
                if SAVE_MODEL:
                    torch.save(model.state_dict(), f"trained_models/{ENV_ID}_{completed_episodes}_EP.pth")
                break
        
        ## If we still haven't filled the batch, restart the loop to get another experience
        if len(batch) < BATCH_SIZE:
            continue
        
        ## Break when we reach the desired number of timesteps
        if steps >= TIMESTEPS:
            print("Timed out!")
            break

        ## Unpack the experiences batch to get the training data
        states_t, actions_t, rewards_t = unpack_batch(batch, model, GAMMA, N_STEPS)
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
        log_prob_v = F.log_softmax(policy_logits_t, dim=1) 

        ## Scale by the advantage
        policy_loss_t = -(advantages_t * log_prob_t).mean() # Minus sign because we want to maximize this function
        log_prob_actions_v = -(advantages_t * log_prob_v[range(BATCH_SIZE), actions_t]).mean()
        
        ## Calculate the entropy loss
        entropy_loss_t = policy_dist.entropy().mean()

        ## Add the losses
        loss_t = COEF_VALUE*loss_value_t + policy_loss_t - BETA_ENTROPY*entropy_loss_t

        ## Backpropagate the loss and update the weights
        loss_t.backward()
        optimizer.step()

    # print(rewards_t.mean())
    
    while True:
        env = gym.make(ENV_ID)
        see_model_in_action(env, model)