import gym
import sys
import math
import random
import numpy as np
import matplotlib.pyplot as plt

episodes_nb = 1000
env = gym.make('CartPole-v0')

episode_rewards     = np.zeros(episodes_nb)
episode_exploration = np.zeros(episodes_nb)

Q = {}
RETURNS = {}

# lambda
discount_factor = 0.8

for episode_i in range(episodes_nb):
	sys.stdout.write("\r" + str(episode_i+1) + "/" + str(episodes_nb))
	sys.stdout.flush()

	observations = env.reset()
	episode = []
	done = False

	while not done:
		state = ''
		for o in observations: state += str(math.floor(o))

		nA = env.action_space.n
		if state not in Q:
			Q[state] = [0]*nA
			RETURNS[state] = {}
			for action in range(nA):
				RETURNS[state][action] = []

		if random.random() < 0.1:
			episode_exploration[episode_i] += 1
			action = env.action_space.sample()
		else:
			action = np.argmax(Q[state])

		observations, reward, done, _ = env.step(action)
		episode.append((state, action, reward))
		episode_rewards[episode_i] += reward

	for state, action, reward in episode:
		firstOccurence = next(i for i, x in enumerate(episode) if x[0] == state and x[1] == action)
		# Discount each rewards
		G = sum(x[2]*discount_factor**i for i, x in enumerate(episode[firstOccurence:]))
		RETURNS[state][action].append(G)
		Q[state][action] = sum(RETURNS[state][action])/len(RETURNS[state][action])


plt.plot(
	range(episodes_nb), episode_rewards,
	range(episodes_nb), episode_exploration
)
plt.ylabel('Reward by episode')
plt.show()
