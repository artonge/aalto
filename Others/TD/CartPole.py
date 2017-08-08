import gym
import numpy as np
import sys
import math
from collections import defaultdict
import matplotlib
import matplotlib.pyplot as plt


def make_epsilon_greedy_policy(Q, epsilon, nA):
	def policy_fn(state):
		A = np.ones(nA, dtype=float) * epsilon / nA
		best_action = np.argmax(Q[state])
		A[best_action] += (1.0 - epsilon)
		return np.random.choice(np.arange(len(A)), p=A)
	return policy_fn


def stringify(obs):
	state = ''
	for o in obs: state += str(math.floor(o))
	return state


def sarsa(env, num_episodes, discount_factor=1.0, alpha=0.5, epsilon=0.1):
	# The final action-value function.
	# A nested dictionary that maps state -> (action -> action-value).
	Q = defaultdict(lambda: np.zeros(env.action_space.n))

	# The policy we're following
	policy = make_epsilon_greedy_policy(Q, epsilon, env.action_space.n)

	for i_episode in range(num_episodes):
		if (i_episode + 1) % 100 == 0:
			print("\rEpisode ", i_episode + 1,"/", num_episodes,".")
			sys.stdout.flush()

		done = False
		t = 0
		obs    = env.reset()
		state  = stringify(obs)
		action = policy(state)

		while not done:
			obs, reward, done, _ = env.step(action)
			nextstate = stringify(obs)
			nextaction = policy(nextstate)

			episode_rewards[i_episode] += reward
			target = reward + discount_factor * Q[nextstate][nextaction]
			Q[state][action] += alpha * (target - Q[state][action])
			print(Q[state][action])
			state  = nextstate
			action = nextaction
			t += 1

	return Q


nbEpisodes = 20000
env = gym.make('CartPole-v0')

# Keeps track of useful statistics
episode_rewards     = np.zeros(nbEpisodes)
episode_exploration = np.zeros(nbEpisodes)

Q = sarsa(env, nbEpisodes)

matplotlib.style.use('ggplot')
plt.plot(
	range(nbEpisodes), episode_rewards,
	range(nbEpisodes), episode_exploration
)
plt.ylabel('Reward by episode')
plt.show()
