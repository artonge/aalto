# Solution to CartPole-v0 using MonteCarlo and an e-greedy policy
import gym
import math
from random import randint
import numpy as np
from collections import defaultdict
import matplotlib
import matplotlib.pyplot as plt

def learn(episodeCount):
	for i_episode in range(episodeCount):
		if (i_episode + 1) % 100 == 0:
			print("\rEpisode ", i_episode + 1,"/", episodeCount,".")

		obs = env.reset()
		episode = []
		done = False

		while not done:
			# Stringify the state
			state = ''
			for o in obs: state += str(math.floor(o))
			# Take step
			action = policy(state, i_episode)
			obs, reward, done, _ = env.step(action)
			episode.append((state, action, reward))

		# Update stats
		# Update Q for each step of the episode
		G = 0
		for i, step in enumerate(reversed(episode)):
			G += step[2]
			s = step[0]
			a = step[1]
			RETURNS[s][a].append(G)
			Q[s][a] = sum(RETURNS[s][a])/len(RETURNS[s][a])
		episode_rewards[i_episode] = G


# e-greedy policy
# Gradually stops exploring
def policy(state, i_episode):
	nA = env.action_space.n
	if state not in Q:
		Q[state] = np.zeros(nA)
		RETURNS[state] = {}
		for a in range(nA):
			RETURNS[state][a] = []
	if randint(0, 100) < 50*0.99**i_episode:
		episode_exploration[i_episode] += 1
		return env.action_space.sample()
	else:
		return np.argmax(Q[state])


nbEpisodes = 1000
env = gym.make('CartPole-v0')

Q = {}
RETURNS = {}

episode_rewards     = np.zeros(nbEpisodes)
episode_exploration = np.zeros(nbEpisodes)

learn(nbEpisodes)

matplotlib.style.use('ggplot')
plt.plot(
	range(nbEpisodes), episode_rewards,
	range(nbEpisodes), episode_exploration
)
plt.ylabel('Reward by episode')
plt.show()
