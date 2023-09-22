import gymnasium as gym
from gymnasium.wrappers import TransformReward

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env

class RewardsLagWrapper(gym.Wrapper):
    def __init__(self, env):
        super(RewardsLagWrapper, self).__init__(env)

    def step(self, action):
        observation, reward, done, info, _ = self.env.step(action)
        
        modified_reward = self.custom_reward_function(observation, reward)
        return observation, modified_reward, done, info, _
    
    def custom_reward_function(self, observation, reward):
        if observation[0] > 2 or observation[0] < -2:
            print('Bonus points')
            reward += 5
        return reward

env = gym.make('CartPole-v1')
env = RewardsLagWrapper(env)

model = PPO('MlpPolicy', env, verbose=1)
model.learn(total_timesteps=200000)

vec_env = make_vec_env('CartPole-v1', n_envs=4)
obs = vec_env.reset()

while True:
    action, _states = model.predict(obs)
    obs, rewards, dones, info = vec_env.step(action)
    vec_env.render("human")
