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
W = {}

alpha = 0.5
discount_factor = 1.0
Lambda = 0.5

for episode_i in range(episodes_nb):
	sys.stdout.write("\r" + str(episode_i+1) + "/" + str(episodes_nb))
	sys.stdout.flush()

	observations = env.reset()
	done = False

	e = 0

	while not done:
		state = ''
		for o in observations: state += str(math.floor(o))

		if state not in Q:
			Q[state] = [0]*env.action_space.n
			W[state] = 0

		if random.random() < 0.1:
			episode_exploration[episode_i] += 1
			action = env.action_space.sample()
		else:
			action = np.argmax(Q[state])

		observations, reward, done, _ = env.step(action)
		episode_rewards[episode_i] += reward

		e = discount_factor * Lambda * e +

		d = reward + discount_factor * np.argmax(Q[nextstate]) - Q[state][action]


plt.plot(
	range(episodes_nb), episode_rewards,
	range(episodes_nb), episode_exploration
)
plt.ylabel('Reward by episode')
plt.show()
