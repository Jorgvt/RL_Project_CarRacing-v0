import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import gym
from cartpole_v1_reinforce_ptan import PGN

env = gym.make("CartPole-v1")

policy = PGN(env.observation_space.shape[0], env.action_space.n)
print("Loading model...")
policy.load_state_dict(torch.load("saved_models/REINFORCE_CartPole-v1.pth"))
print("Model loaded!")

# while True:
done = False
obs = env.reset()
while not done:
    env.render()
    obs = np.expand_dims(obs, 0) # Give a batch dim
    obs = torch.tensor(obs).float() # Turn into float tensor

    with torch.no_grad():
        actions_prob = policy(obs)
        actions_prob = F.softmax(actions_prob, dim=1) # Turn logits into probs
        actions_prob = actions_prob[0].cpu().numpy() # Transform to numpy

    ## Sample from the actions distributions to obtain the action 
    action = np.random.choice(len(actions_prob), p=actions_prob)

    obs, reward, done, info = env.step(action)