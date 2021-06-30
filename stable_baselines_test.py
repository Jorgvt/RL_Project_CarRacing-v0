import gym
from stable_baselines3 import A2C

env_id = 'CartPole-v1'
video_folder = 'logs/videos/'
video_length = 100

env = gym.make(env_id)

model = A2C('MlpPolicy', env, verbose=1)
model.learn(total_timesteps=10000)

obs = env.reset()

for i in range(1000):
    action, _state = model.predict(obs, deterministic=True)
    obs, reward, done, info = env.step(action)
    env.render()
    if done:
        obs = env.reset()

# Save the video
env.close()