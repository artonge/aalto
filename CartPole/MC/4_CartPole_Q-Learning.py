import gym
import sys
import math
import random
import numpy as np
import matplotlib.pyplot as plt

episodes_nb = 1500
env = gym.make('CartPole-v0')

episode_rewards     = np.zeros(episodes_nb)
episode_exploration = np.zeros(episodes_nb)

Q = {}
RETURNS = {}


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

		# The first two third, learn with a radom policy, then exploit with a greedy policy
		if episode_i < 2.0/3.0 * episodes_nb:
			action = env.action_space.sample()
			# The random policy could also be an e-greedy policy/ See bellow:
			# if random.random() < 0.7:
			# 	action = env.action_space.sample()
			# else:
			# 	action = np.argmax(Q[state])
			observations, reward, done, _ = env.step(action)
			episode.append((state, action, reward))
			episode_rewards[episode_i] += reward
		else:
			action = np.argmax(Q[state])
			observations, reward, done, _ = env.step(action)
			episode_rewards[episode_i] += reward

	# Stop learning after the two third of episodes
	if episode_i < 2.0/3.0 * episodes_nb:
		for state, action, reward in episode:
			firstOccurence = next(i for i, x in enumerate(episode) if x[0] == state)
			G = sum(x[2] for i, x in enumerate(episode[firstOccurence:]))
			RETURNS[state][action].append(G)
			Q[state][action] = sum(RETURNS[state][action])/len(RETURNS[state][action])


plt.plot(
	range(episodes_nb), episode_rewards,
	range(episodes_nb), episode_exploration
)
plt.ylabel('Reward by episode')
plt.show()
