import gymnasium as gym
from gymnasium.wrappers import TransformReward

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy

import csv


class CustomRewardWrapper(gym.Wrapper):
    def __init__(self, env):
        super(CustomRewardWrapper, self).__init__(env)

        # flags bugs found during one episode
        self.found_bug_one = False
        self.found_bug_two = False

        # tracks progress across episodes
        self.episode_count = 0
        self.bug_one_count = 0
        self.bug_two_count = 0
        self.found_both = 0
        
        #self.explore_list = [False] * 5

    def step(self, action):

        observation, reward, terminated, truncated, info = self.env.step(action)

        if terminated or truncated:
            
            self.episode_count += 1
            if self.found_bug_one and self.found_bug_two:
                self.found_both += 1

            self.found_bug_one = False
            self.found_bug_two = False

            #self.explore_list = [False] * 5
        
        modified_reward = self.custom_reward_function(observation, reward)
        return observation, modified_reward, terminated, truncated, info
    
    # reward if a bug has been found for the first time
    def custom_reward_function(self, observation, reward):

        # check if cart is in first bug area and not already found
        if not self.found_bug_one and observation[0] > 0.45 and observation[0] < 0.5:
            self.bug_one_count += 1

            #print('found bug one')
            reward += 50
            self.found_bug_one = True

        # check if cart is in first bug area and not already found
        if not self.found_bug_two and observation[0] > -0.5 and observation[0] < -0.45:
            self.bug_two_count += 1

            #print('found bug two')
            reward += 50
            self.found_bug_two = True

        #testing curiosity
        #reward = self.curiosity_reward_function(observation, reward)
        
        return reward
    

    def curiosity_reward_function(self, observation, reward):

        if observation[0] > -1.1 and observation[0] < -0.9 and self.explore_list[4] == False:
            self.explore_list[4] = True
            reward += 50
        
        if observation[0] > 0.9 and observation[0] < 1.1 and self.explore_list[0] == False:
            self.explore_list[0] = True
            reward += 50

        if observation[0] > 1.9 and observation[0] < 2.1 and self.explore_list[1] == False:
            self.explore_list[1] = True
            reward += 50

        if observation[0] > 2.9 and observation[0] < 3.1 and self.explore_list[2] == False:
            self.explore_list[2] = True
            reward += 50

        if observation[0] > 4.9 and observation[0] < 4.1 and self.explore_list[3] == False:
            self.explore_list[3] = True
            reward += 50

        return reward

for i in range(10):
    
    ## TRAINING ###
    env = gym.make('CartPole-v1')
    env = CustomRewardWrapper(env)

    training_timesteps = 100000
    model = PPO('MlpPolicy', env, verbose=1)
    model.learn(total_timesteps=training_timesteps)

    env.close()

    ## EVALUATION ##

    eval_env = gym.make('CartPole-v1')
    eval_env = CustomRewardWrapper(env)

    mean_reward, std_reward = evaluate_policy(
        model,
        eval_env,
        n_eval_episodes=100,
        deterministic=True,
    )

    # ouput stats
    print(f'train_timesteps={training_timesteps}, eval_episodes={eval_env.episode_count}, bug_one_count={eval_env.bug_one_count}, bug_two_count={eval_env.bug_two_count}, found_both_count={eval_env.found_both}')
    print(f"mean_reward={mean_reward:.2f} +/- {std_reward}")

    with open('CartPoleRELINEResults.csv', 'a', newline='\n') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow([eval_env.episode_count, eval_env.bug_one_count, eval_env.bug_two_count, eval_env.found_both])

    eval_env.close()
    

# just plays with rendered screen
#vec_env = make_vec_env('CartPole-v1', n_envs=4)
#obs = vec_env.reset()

#while True:
#    action, _states = model.predict(obs)
#    obs, rewards, dones, info = vec_env.step(action)
#    vec_env.render("human")
