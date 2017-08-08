# Solution to FrozenLake-v0 using MonteCarlo and an e-greedy policy
import gym
from random import randint
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from collections import defaultdict

def learn(episodeCount):
	returns_sum   = defaultdict(float)
	returns_count = defaultdict(float)

	for i_episode in range(episodeCount):
		state   = env.reset()
		episode = []
		done    = False

		while not done:
			# Take step
			action = policy(state, i_episode)
			state, reward, done, _ = env.step(action)

			if done:
				reward = 1.0 if reward > 0.0 else -1.0
			else:
				reward = -0.01

			episode_rewards[i_episode] += reward
			episode.append((state, action, reward))

		# Update Q for each step of the episode
		for i, step in enumerate(reversed(episode)):
			s = step[0]
			a = step[1]
			returns_count[(s, a)] += 1
			returns_sum[(s, a)]   += step[2]
			Q[s][a] = returns_sum[(s, a)] / returns_count[(s, a)]

	print(returns_sum)

# e-greedy policy
# Gradually stops exploring
def policy(state, i_episode):
	# return env.action_space.sample()
	if randint(0, 100) < 50*0.99**i_episode:
		return env.action_space.sample()
	else:
		return np.argmax(Q[state])


nbEpisodes = 3000
env = gym.make('FrozenLake-v0')

episode_rewards = np.zeros(nbEpisodes)

Q = defaultdict(lambda: np.zeros(env.action_space.n))

learn(nbEpisodes)
print(Q)
matplotlib.style.use('ggplot')
plt.plot(range(nbEpisodes), episode_rewards)
plt.ylabel('Reward by episode')
plt.show()
