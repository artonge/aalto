import gym
import matplotlib
import numpy as np
import sys
from collections import defaultdict
import plotting


def make_epsilon_greedy_policy(Q, epsilon, nA):
	def policy_fn(state):
		A = np.ones(nA, dtype=float) * epsilon / nA
		best_action = np.argmax(Q[state])
		A[best_action] += (1.0 - epsilon)
		return A
	return policy_fn


def stringify(obs):
	state = ''
	for o in obs: state += str(int(o*100))
	return state


def sarsa(env, num_episodes, discount_factor=1.0, alpha=0.5, epsilon=0.1):
	# The final action-value function.
	# A nested dictionary that maps state -> (action -> action-value).
	Q = defaultdict(lambda: np.zeros(env.action_space.n))

	# Keeps track of useful statistics
	stats = plotting.EpisodeStats(
		episode_lengths=np.zeros(num_episodes),
		episode_rewards=np.zeros(num_episodes)
	)

	# The policy we're following
	policy = make_epsilon_greedy_policy(Q, epsilon, env.action_space.n)

	for i_episode in range(num_episodes):
		if (i_episode + 1) % 100 == 0:
			print("\rEpisode {}/{}.".format(i_episode + 1, num_episodes), end="")
			sys.stdout.flush()

		done = False
		t = 0
		obs  = env.reset()
		state = stringify(obs)
		props  = policy(state)
		action = np.random.choice(np.arange(len(props)), p=props)
		while not done:
			obs, reward, done, _ = env.step(action)
			nextstate = stringify(obs)
			props = policy(nextstate)
			nextaction = np.random.choice(np.arange(len(props)), p=props)

			stats.episode_rewards[i_episode] += reward
			stats.episode_lengths[i_episode] = t

			Q[state][action] += alpha * (reward + discount_factor * Q[nextstate][nextaction] - Q[state][action])

			state  = nextstate
			action = nextaction
			t += 1

	return Q, stats


env = gym.make('CartPole-v0')

Q, stats = sarsa(env, 20000)

matplotlib.style.use('ggplot')
plotting.plot_episode_stats(stats)
