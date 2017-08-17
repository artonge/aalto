import gym
import sys
import math
import random
import numpy as np
import matplotlib.pyplot as plt

episodes_nb = 100000
env = gym.make('CartPole-v0')

episode_rewards     = np.zeros(episodes_nb)
episode_exploration = np.zeros(episodes_nb)

Q = {}

alpha = 0.1
discount_factor = 1.0

for episode_i in range(episodes_nb):
	sys.stdout.write("\r" + str(episode_i+1) + "/" + str(episodes_nb))
	sys.stdout.flush()

	# Get first step, and first random action
	observations = env.reset()
	state = ''
	for o in observations: state += str(math.floor(o))
	if state not in Q:
		Q[state] = [0]*env.action_space.n

	action = env.action_space.sample()

	done = False

	while not done:
		observations, reward, done, _ = env.step(action)
		nextstate = ''
		for o in observations: nextstate += str(math.floor(o))

		if nextstate not in Q:
			Q[nextstate] = [0]*env.action_space.n

		if random.random() < 0.1:
			episode_exploration[episode_i] += 1
			nextaction = env.action_space.sample()
		else:
			nextaction = np.argmax(Q[nextstate])

		episode_rewards[episode_i] += reward

		# Update Q
		target = reward + discount_factor*Q[nextstate][nextaction]
		Q[state][action] += alpha * (target - Q[state][action])

		state  = nextstate
		action = nextaction


plt.plot(
	range(episodes_nb), episode_rewards,
	range(episodes_nb), episode_exploration
)
plt.ylabel('Reward by episode')
plt.show()
