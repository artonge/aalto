import gym
import matplotlib
import numpy as np
from collections import defaultdict
from BlackjackEnv import BlackjackEnv
import plotting


def mc_prediction(policy, env, num_episodes, discount_factor=1.0):
	returns_sum = defaultdict(float)
	returns_count = defaultdict(float)

	V = defaultdict(float)

	for i in range(num_episodes):
		state = env.reset()
		done = False
		episode = []

		# Generate episode
		while not done:
			action = sample_policy(state)
			next_state, reward, done, _ = env.step(action)
			episode.append((state, action, reward))
			state = next_state

		# Compute states values
		for state, action, reward in episode:
			firstOccurence = next(i for i, x in enumerate(episode) if x[0] == state)
			G = sum([x[2]*discount_factor**i for i, x in enumerate(episode[firstOccurence:])])
			returns_sum[state] += G
			returns_count[state] += 1.0
			V[state] = returns_sum[state]/returns_count[state]
			break

	return V


def sample_policy(observation):
	score, dealer_score, usable_ace = observation
	return 0 if score >= 20 else 1


matplotlib.style.use('ggplot')
env = BlackjackEnv()


V_10k = mc_prediction(sample_policy, env, num_episodes=10000)
plotting.plot_value_function(V_10k, title="10,000 Steps")

V_500k = mc_prediction(sample_policy, env, num_episodes=500000)
plotting.plot_value_function(V_500k, title="500,000 Steps")
