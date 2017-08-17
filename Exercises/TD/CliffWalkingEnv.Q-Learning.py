import gym
import matplotlib
import numpy as np
import sys
from collections import defaultdict
from CliffWalkingEnv import CliffWalkingEnv
import plotting


def make_epsilon_greedy_policy(Q, epsilon, nA):
	def policy_fn(observation):
		A = np.ones(nA, dtype=float) * epsilon / nA
		best_action = np.argmax(Q[observation])
		A[best_action] += (1.0 - epsilon)
		return np.random.choice(np.arange(len(A)), p=A)
	return policy_fn

def make_random_policy(nA):
	A = np.ones(nA, dtype=float) / nA
	def policy_fn():
		return np.random.choice(np.arange(len(A)), p=A)
	return policy_fn


def q_learning(env, num_episodes, discount_factor=1.0, alpha=0.5, epsilon=0.1):
	# The final action-value function.
	# A nested dictionary that maps state -> (action -> action-value).
	Q = defaultdict(lambda: np.zeros(env.action_space.n))

	# Keeps track of useful statistics
	stats = plotting.EpisodeStats(
		episode_lengths=np.zeros(num_episodes),
		episode_rewards=np.zeros(num_episodes)
	)

	# The policy we're following
	behavior_policy = make_random_policy(env.action_space.n)
	target_policy   = make_epsilon_greedy_policy(Q, epsilon, env.action_space.n)

	for i_episode in range(num_episodes):
		# Print out which episode we're on, useful for debugging.
		if (i_episode + 1) % 100 == 0:
			print("\rEpisode {}/{}.".format(i_episode + 1, num_episodes), end="")
			sys.stdout.flush()

		done   = False
		state  = env.reset()
		t = 0
		while not done:
			action = target_policy(state)
			nextstate, reward, done, _ = env.step(action)

			stats.episode_rewards[i_episode] += reward
			stats.episode_lengths[i_episode]  = t

			Q[state][action] += alpha * (reward + discount_factor * np.max(Q[nextstate]) - Q[state][action])

			state  = nextstate

			t += 1

	return Q, stats


env = CliffWalkingEnv()

Q, stats = q_learning(env, 500)

matplotlib.style.use('ggplot')
plotting.plot_episode_stats(stats)
