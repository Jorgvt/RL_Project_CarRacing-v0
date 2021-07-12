import numpy as np
import gym
import ptan
import torch
import torch.nn as nn
import torch.nn.functional as F

GAMMA = 0.99
LEARNING_RATE = 0.001
ENTROPY_BETA = 0.01 # Scale of the entropy bonus
BATCH_SIZE = 8
REWARD_STEPS = 10 # How many steps ahead the Bellman equation is unrolled to estimate the discounted reward
SAVE_MODEL = True

## Define the policy network
class PGN(nn.Module):
    def __init__(self, input_size, n_actions):
        super(PGN, self).__init__()
        self.net = nn.Sequential(*[
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.Linear(128, n_actions)
        ])
    
    ## Despite the fact our network returns probabilities, we are not applying softmax to the last layer
    ## because we will use the log_softmax function to calculate the log of the softmax at once.
    ## This method is much more numerically stable.

    def forward(self, x):
        return self.net(x)

def calc_qvals(rewards):
    """
    Accepts a list of rewards for the whole episode and needs to calculate the discounted total reward for 
    every step. To tdo this efficiently, we calculate the rewards from the end of the local reward list.
    The last step of the epiosde will have a total reward equal to its local reward. The step before will
    have the total reward of r_{t-1} + GAMMA*r_{t}.
    sum_r contains the total reward for the previous steps, so we need to multiply sum_r by gamma and sum
    the local reward.
    """
    res = []
    sum_r = 0.0
    for r in reversed(rewards):
        sum_r *= GAMMA
        sum_r += r
        res.append(sum_r)
    return list(reversed(res))

if __name__ == "__main__":
    ## Initialize the environmnet
    env = gym.make("CartPole-v0")
    
    ## Initialize the network with random weights
    policy = PGN(env.observation_space.shape[0], env.action_space.n)
    optimizer = torch.optim.Adam(policy.parameters(), lr=LEARNING_RATE)

    ## Initialize PTAN's agent and exp_source
    agent = ptan.agent.PolicyAgent(policy,
                                   preprocessor=ptan.agent.float32_preprocessor,
                                   apply_softmax=True)
    exp_source = ptan.experience.ExperienceSourceFirstLast(env,
                                                           agent,
                                                           gamma=GAMMA,
                                                           steps_count=REWARD_STEPS) # steps_count tells how many steps to unroll the Bellman equation

    ## Variables used for reporting
    total_rewards = []
    done_episodes = 0

    ## Variables to gather training data
    batch_episodes = 0
    reward_sum = 0 # Will be used as baseline
    batch_states, batch_actions, batch_scales = [], [], []

    for step_idx, exp in enumerate(exp_source):
        ## Keep the sum of the discounted reward to calculate the baseline
        reward_sum += exp.reward
        baseline = reward_sum / (step_idx+1)

        batch_states.append(exp.state)
        batch_actions.append(int(exp.action))
        batch_scales.append(exp.reward - baseline)
        
        ## This part is performed at the end of the episode as is responsible for reporting
        ## exp_source only stores total_rewards when an episode is finished, so 
        ## pop_total_rewards() only returns something when the episode is finished too.
        new_rewards = exp_source.pop_total_rewards()
        if new_rewards:
            done_episodes += 1
            reward = new_rewards[0]
            total_rewards.append(reward)
            mean_rewards = float(np.mean(total_rewards[-100:]))
            print(f"Step {step_idx}: Reward {reward}, Mean_100: {mean_rewards}, Episodes: {done_episodes}")

            if mean_rewards > 195: # Solved environment condition
                print(f"Solved in {step_idx} steps and {done_episodes} episodes!")
                if SAVE_MODEL:
                    torch.save(policy.state_dict(), "saved_models/REINFORCE_PG_CartPole-v0.pth")
                break
        
        ## We don't need to wait until the episodes are over to train,
        ## we only need to wait until we fill a batch.
        if len(batch_scales) < BATCH_SIZE:
            continue
        
        ## When enough episodes have passed since the last training step, we perform optimization on the gathered examples.
        ## The first step is converting everything into PyTorch.
        optimizer.zero_grad()
        states_v = torch.FloatTensor(batch_states)
        batch_actions_t = torch.LongTensor(batch_actions)
        batch_scales_v = torch.FloatTensor(batch_scales)

        logits_v = policy(states_v)
        log_prob_v = F.log_softmax(logits_v, dim=1)
        # It's important to use the action that was used during the generation of the training data, 
        # because it was not the max prob but a sampled action from the output distribution.
        log_prob_actions_v = batch_scales_v * log_prob_v[range(BATCH_SIZE), batch_actions_t] # Indexing magics
        loss_policy_v = -log_prob_actions_v.mean()

        ## Entropy calculation
        prob_v = F.softmax(logits_v, dim=1)
        entropy_v = -(prob_v * log_prob_v).sum(dim=1).mean()
        entropy_loss_v = -ENTROPY_BETA * entropy_v

        ## Calculate the full loss
        loss_v = loss_policy_v + entropy_loss_v

        loss_v.backward()
        optimizer.step()

        ## Calculate the KL divergence between the new and the old policies
        new_logits_v = policy(states_v)
        new_prob_v = F.softmax(new_logits_v, dim=1)
        kl_div_v = -((new_prob_v / prob_v).log()*prob_v).sum(dim=1).mean()
        # print(f"KL -> {kl_div_v}")
        ## Reset the episodes counter and clear the lists
        batch_episodes = 0
        batch_states.clear()
        batch_actions.clear()
        batch_scales.clear()

