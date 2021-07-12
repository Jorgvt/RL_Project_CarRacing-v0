import numpy as np
import gym
import torch
import torch.nn as nn
import torch.nn.functional as F

GAMMA = 0.99
LEARNING_RATE = 0.01
EPISODES_TO_TRAIN = 4 # How many complete episodes we will use for training

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

    ## Play N full episodes, saving their (s,a,r,s') transitions
    for episode_i in range(EPISODES_TO_TRAIN):
        obs = env.reset()
        
        ## Play each episode until the done flag is returned as True
        done = False
        while done:
            sars = {'state':obs}

            obs = np.expand_dims(obs, 0) # Give a batch dim
            obs = torch.tensor(obs).float() # Turn into float tensor

            with torch.no_grad():
                actions_prob = policy(obs)
                actions_prob = F.softmax(actions_prob) # Turn logits into probs
                actions_prob = actions_prob[0].cpu().numpy() # Transform to numpy

            ## Sample from the actions distributions to obtain the action 
            action = np.random.choice(len(actions_prob), p=actions_prob)
            obs, reward, done, info = env.step(action)

            sars['action'] = action
            sars['reward'] = reward
            sars['next_state'] = obs
