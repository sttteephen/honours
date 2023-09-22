import gymnasium as gym
from gymnasium.wrappers import TransformReward

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env

episode_count = 0
bug_one_count = 0
bug_two_count = 0
found_both = 0

class RewardLagWrapper(gym.Wrapper):
    def __init__(self, env):
        super(RewardLagWrapper, self).__init__(env)
        self.found_reward_one = False
        self.found_reward_two = False

    def step(self, action):
        global episode_count, found_both

        observation, reward, terminated, truncated, info = self.env.step(action)

        if terminated or truncated:
            
            episode_count += 1
            if self.found_reward_one and self.found_reward_two:
                found_both += 1

            self.found_reward_one = False
            self.found_reward_two = False
        
        modified_reward = self.custom_reward_function(observation, reward)
        return observation, modified_reward, terminated, truncated, info
    
    # reward if a bug has been found for the first time
    def custom_reward_function(self, observation, reward):

        # check if cart is in first bug area and not already found
        if observation[0] > 1 and observation[0] < 1.5 and self.found_reward_one == False:
            global bug_one_count
            bug_one_count += 1

            print('found bug one')
            reward += 50
            self.found_reward_one = True

        # check if cart is in first bug area and not already found
        if observation[0] > -1.5 and observation[0] < -1 and self.found_reward_two == False:
            global bug_two_count
            bug_two_count += 1

            print('found bug two')
            reward += 50
            self.found_reward_two = True
        
        return reward

# make wrapped env then train PPO
env = gym.make('CartPole-v1')
env = RewardLagWrapper(env)

model = PPO('MlpPolicy', env, verbose=1)
model.learn(total_timesteps=100000)

# ouput stats
print(episode_count, bug_one_count, bug_two_count)
input()

# just plays with rendered screen
vec_env = make_vec_env('CartPole-v1', n_envs=4)
obs = vec_env.reset()

while True:
    action, _states = model.predict(obs)
    obs, rewards, dones, info = vec_env.step(action)
    vec_env.render("human")
