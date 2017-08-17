import gym
import sys
import math
import random
import numpy as np
import matplotlib.pyplot as plt

import gym_sudoku

episodes_nb = 1
env = gym.make('Sudoku-v0')

episode_rewards     = np.zeros(episodes_nb)
episode_exploration = np.zeros(episodes_nb)

Q = {}
RETURNS = {}
A = {}


for episode_i in range(episodes_nb):
	sys.stdout.write("\r" + str(episode_i+1) + "/" + str(episodes_nb))
	sys.stdout.flush()

	observations = env.reset()
	episode = []
	done = False
	t = 0
	while not done:
		state = str(observations)
		if t % 1000 == 0:
			env.render()


		if state not in Q:
			Q[state] = {}
			RETURNS[state] = {}

		# 10% chance to not take the best action
		a = np.argmax(Q[state])
		if random.random() < 0.1 or not a:
			episode_exploration[episode_i] += 1
			a = env.action_space.sample()

		action = str(a)
		A[action] = a

		observations, reward, done, _ = env.step(A[action])
		episode.append((state, action, reward))
		episode_rewards[episode_i] += reward

		t += 1

	for state, action, reward in episode:
		firstOccurence = next(i for i, x in enumerate(episode) if x[0] == state)
		G = sum(x[2] for i, x in enumerate(episode[firstOccurence:]))
		if action not in RETURNS[state]:
			RETURNS[state][action] = []
		RETURNS[state][action].append(G)
		Q[state][action] = sum(RETURNS[state][action])/len(RETURNS[state][action])


plt.plot(
	range(episodes_nb), episode_rewards,
	range(episodes_nb), episode_exploration
)
plt.ylabel('Reward by episode')
plt.show()
