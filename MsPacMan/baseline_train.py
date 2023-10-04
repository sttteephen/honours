
import gymnasium as gym
from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy
import numpy as np
import os
import cv2
from PIL import Image
from skimage.metrics import structural_similarity as ssim


# bug parameters folder_location, left, top, right, bottom
BUG1 = ('bug_left/', 0, 90, 15, 120, 'bug1')
BUG2 = ('bug_right/', 305, 90, 320, 120, 'bug2')


def check_bug(folder, left, top, right, bottom, name):
    folder_bug = folder
    files = os.listdir(folder_bug)
    img_bug = [file for file in files if file.startswith('bug')]
    img = Image.open("current_screen.png")
    im1 = img.crop((left, top, right, bottom))
    im1.save('current_test.png')
    imgA = cv2.imread("current_test.png")
    for elem in img_bug:
        imgB = cv2.imread(folder_bug + elem)
        imgA = np.squeeze(imgA)
        imgB = np.squeeze(imgB)
        s = ssim(imgA, imgB, channel_axis=2)
        if s > 0.9:
            #print(name, s)
            return True
    return False


class CustomRewardWrapper(gym.Wrapper):
    def __init__(self, env):
        super(CustomRewardWrapper, self).__init__(env)

        self.bug_flags = [False, False]
        self.ep_count = 0
        self.one_bug = 0
        self.two_bug = 0

    def step(self, action):
        observation, reward, terminated, truncated, info = self.env.step(action)
        
        self.env.ale.saveScreenPNG('current_screen.png')

        if not self.bug_flags[0] and check_bug(*BUG1):
            self.bug_flags[0] = True
        if not self.bug_flags[1] and check_bug(*BUG2):
            self.bug_flags[1] = True

        if terminated or truncated:
            self.ep_count += 1

            if self.bug_flags[0] and self.bug_flags[1]:
                self.one_bug += 1
                self.two_bug += 1
            elif self.bug_flags[0] or self.bug_flags[1]:
                self.one_bug += 1

            self.bug_flags = [False, False]

        return observation, reward, terminated, truncated, info

## TRAINING FROM SCRATCH OF LOAD MODEL ##
def train():
    env = gym.make('MsPacman-v4')

    #model = DQN('CnnPolicy', env, buffer_size=10000, verbose=1)

    model = DQN.load('latest_mspacman_baseline')
    model.set_env(env)

    model.learn(total_timesteps=500000, tensorboard_log="./tensor_baseline/")
    model.save('latest_mspacman_baseline')


## EVALUATION ##
def evaluate():
    model = DQN.load('latest_mspacman_baseline')
    env = gym.make('MsPacman-v4')
    env = CustomRewardWrapper(env) # wrapped so it counts the bugs

    mean_reward, std_reward = evaluate_policy(
        model,
        env,
        n_eval_episodes=100,
        deterministic=True,
    )

    print(f'episodes={env.ep_count}, one_bug={env.one_bug}, two_bugs={env.two_bug}')


## TESTING CODE HUMAN RENDER ##
def test():
    env = gym.make('MsPacman-v4', render_mode='human')
    model = DQN.load('latest_mspacman_baseline')
    model.set_env(env)

    for game in range(10):
        state = env.reset()

        # do nothing for 65 timesteps at start of the game
        for i in range(65):
            state, reward, terminated, truncated, info = env.step(0)

        while True:

            action, _states = model.predict(state, deterministic=True)
            state, reward, terminated, truncated, info = env.step(action)
            
            if terminated or truncated:
                break

evaluate()