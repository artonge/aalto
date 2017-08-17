import gym
import sys
import math
from random import randint
import numpy as np
import matplotlib.pyplot as plt


episodes_nb = 1000
env = gym.make('CartPole-v0')

episode_rewards = np.zeros(episodes_nb)

# Q will save the value for each state-action pair
Q = {}
# Return will save the experienced return for each states
RETURNS = {}


for episode_i in range(episodes_nb):
	sys.stdout.write("\r" + str(episode_i+1) + "/" + str(episodes_nb))
	sys.stdout.flush()

	observations = env.reset()
	episode = []
	done = False

	while not done:
		# Discretise and stringify the observations
		state = ''
		for o in observations: state += str(math.floor(o))

		# Create the state in Q and RETURNS if needed
		nA = env.action_space.n
		if state not in Q:
			Q[state] = [0]*nA
			RETURNS[state] = {}
			for action in range(nA):
				RETURNS[state][action] = []

		# Get best action for the state
		action = np.argmax(Q[state])
		observations, reward, done, _ = env.step(action)
		# Save step
		episode.append((state, action, reward))
		episode_rewards[episode_i] += reward

	# Update Q for each step of the episode
	for state, action, reward in episode:
		# Get the first occurence of the state
		firstOccurence = next(i for i, x in enumerate(episode) if x[0] == state)
		# Sum all reward following that occurence
		G = sum(x[2] for i, x in enumerate(episode[firstOccurence:]))
		RETURNS[state][action].append(G)
		# ThSet the value to the average of all returns across all steps
		Q[state][action] = sum(RETURNS[state][action])/len(RETURNS[state][action])


plt.plot(range(episodes_nb), episode_rewards)
plt.ylabel('Reward by episode')
plt.show()
