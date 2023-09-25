import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy

# Create the game environment
env = gym.make('MsPacman-v0')

# Wrap it in a vectorized environment
env = DummyVecEnv([lambda: env])

# Instantiate the agent
model = PPO("CnnPolicy", env, verbose=1)

# Train the agent
model.learn(total_timesteps=1000)

# Evaluate the trained agent
mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=10)

print(f"Mean reward: {mean_reward}, Std reward: {std_reward}")

# Save the model
#model.save("ppo_ms_pacman")

# Load the model if needed
# model = PPO.load("ppo_ms_pacman")

# Enjoy the trained agent
obs = env.reset()
for i in range(1000):
    action, _states = model.predict(obs)
    obs, rewards, dones, info = env.step(action)
    env.render()
env.close()