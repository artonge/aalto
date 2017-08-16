import gym
import matplotlib.pyplot as plt
import sys


# The number of episode we will do
episodes_nb = 1000

# Create the environment using gym
env = gym.make('Sudoku-v0')
print env.action_space
print env.observation_space

# Create an array to store the reward for each episode
# Usefull to visalize the learning progress
episode_rewards = [0]*episodes_nb


# Learn
for episode_i in range(episodes_nb):
	# Print progress
	sys.stdout.write("\r" + str(episode_i+1) + "/" + str(episodes_nb))
	sys.stdout.flush()

	obs = env.reset()
	done = False

	while not done:
		print obs
		# Choose random action
		action = env.action_space.sample()
		print action
		# Take action
		obs, reward, done, _ = env.step(action)

		# Update stats
		episode_rewards[episode_i] += reward


# Display graph
plt.plot(range(episodes_nb), episode_rewards)
plt.ylabel('Reward by episode')
plt.show()
