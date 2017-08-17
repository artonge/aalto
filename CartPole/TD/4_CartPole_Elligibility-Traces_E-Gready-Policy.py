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
# Contains the traces for each states
Z = {}

alpha = 0.8
discount_factor = 0.9
Lambda = 0.7

for episode_i in range(episodes_nb):
	sys.stdout.write("\r" + str(episode_i+1) + "/" + str(episodes_nb))
	sys.stdout.flush()

	observations = env.reset()
	state = ''
	for o in observations: state += str(math.floor(o))
	if state not in Q:
		Q[state] = np.zeros(env.action_space.n)
		Z[state] = np.zeros(env.action_space.n)

	done = False

	while not done:
		if random.random() < 0.1:
			episode_exploration[episode_i] += 1
			action = env.action_space.sample()
		else:
			action = np.argmax(Q[state])

		observations, reward, done, _ = env.step(action)
		nextstate = ''
		for o in observations: nextstate += str(math.floor(o))
		if nextstate not in Q:
			Q[nextstate] = np.zeros(env.action_space.n)
			Z[nextstate] = np.zeros(env.action_space.n)

		episode_rewards[episode_i] += reward

		target = reward + discount_factor * np.argmax(Q[nextstate]) - Q[state][action]
		Z[state][action] = 1
		for s in Q:
			Q[s] += alpha * target * Z[s]
			Z[s] *= Lambda

		state = nextstate


plt.plot(
	range(episodes_nb), episode_rewards,
	range(episodes_nb), episode_exploration
)
plt.ylabel('Reward by episode')
plt.show()
