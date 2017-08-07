import gym
from random import randint
import math
import matplotlib
import plotting
import numpy as np

def learn(episodeCount):
	for i_episode in range(episodeCount):
		if (i_episode + 1) % 100 == 0:
			print("\rEpisode {}/{}.".format(i_episode + 1, episodeCount), end="")

		obs = env.reset()
		episode = []
		done = False

		while not done:
			state = ''
			for o in obs: state += str(math.floor(o))

			action = policy(state, i_episode)
			obs, reward, done, _ = env.step(action)

			episode.append((state, action))

			stats.episode_rewards[i_episode] += reward
			stats.episode_lengths[i_episode] += 1

		for i, step in enumerate(episode):
			G = stats.episode_rewards[i_episode]-i
			s = step[0]
			a = step[1]
			Q[s][a] += G
			COUNT[s][a] += 1


# @param state <string>
# @param decay <int>
# @return an action
# The policy progressivly stops exploration and gets greedy
def policy(state, i_episode):
	epsilon = 0.1
	nA = env.action_space.n
	if state not in Q:  # If state does not existe, create it
		Q[state]     = np.zeros(nA)
		COUNT[state] = np.zeros(nA)

	A = np.ones(nA, dtype=float) * epsilon / nA
	best_action = np.argmax(Q[state])
	A[best_action] += (1.0 - epsilon)
	action = np.random.choice(np.arange(len(A)), p=A)
	if action != best_action:
		stats.episode_exploration[i_episode] += 1
	return np.random.choice(np.arange(len(A)), p=A)

	maxA = np.argmax(Q[state]/COUNT[state])
	decay = 15*0.99**i_episode
	# if randint(0, 100) < decay and i_episode < 300:
	if randint(0, 100) < 10:
		stats.episode_exploration[i_episode] += 1
		return env.action_space.sample()
	else:
		return maxA


env = gym.make('CartPole-v0')
nbEpisodes = 500

Q = {}
COUNT = {}

stats = plotting.EpisodeStats(
	episode_lengths=np.zeros(nbEpisodes),
	episode_rewards=np.zeros(nbEpisodes),
	episode_exploration=np.zeros(nbEpisodes)
)

learn(nbEpisodes)

env.close()

matplotlib.style.use('ggplot')
plotting.plot_episode_stats(stats)
