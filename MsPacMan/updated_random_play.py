import gymnasium as gym
import numpy as np
import os
import cv2
from PIL import Image
from skimage.metrics import structural_similarity as ssim


def check_bug1():
    folder_bug = 'bug_left/'
    files = os.listdir(folder_bug)
    img_bug = [file for file in files if file.startswith('bug')]
    img = Image.open("current_screen.png")
    left = 0
    top = 90
    right = 15
    bottom = 120
    im1 = img.crop((left, top, right, bottom))
    im1.save('current_test.png')
    imgA = cv2.imread("current_test.png")
    for elem in img_bug:
        imgB = cv2.imread(folder_bug + elem)
        imgA = np.squeeze(imgA)
        imgB = np.squeeze(imgB)
        s = ssim(imgA, imgB, channel_axis=2)
        if s > 0.9:
            print(s)
            return True
    return False


env = gym.make('MsPacman-v4', render_mode='human')

for game in range(2):
    print('new game')
    state = env.reset()

    for i in range(65):
        state, reward, terminated, truncated, info = env.step(0)

    while True:
        
        # random action
        state, reward, terminated, truncated, info = env.step(env.action_space.sample())
        
        # check injected bugs
        env.env.ale.saveScreenPNG('current_screen.png')
        check_bug1()
        
        print(terminated, truncated)
        if terminated or truncated:
            break
