import gym
import matplotlib
import numpy as np
from collections import defaultdict
import plotting
import sys
import matplotlib.pyplot as plt
from gym.wrappers.monitoring import Monitor



def make_epsilon_greedy_policy(Q, epsilon, nA):
	def policy_fn(observation, i):
		lr = np.exp(alpha*i)
		if np.random.rand() < lr*epsilon:
			return np.random.randint(env.action_space.n)
		else:
			return np.argmax(Q[observation] )
		# A = np.ones(env.action_space.n) * epsilon / nA
		# A[np.argmax(Q[observation])] += 1 - epsilon
		# return np.random.choice(np.arange(len(A)), p=A)

	return policy_fn


def mc_control_epsilon_greedy(env, num_episodes, discount_factor=1.0, epsilon=0.1):
	# Keeps track of sum and count of returns for each state
	# to calculate an average. We could use an array to save all
	# returns (like in the book) but that's memory inefficient.
	returns_sum = defaultdict(float)
	returns_count = defaultdict(float)

	# The final action-value function.
	# A nested dictionary that maps state -> (action -> action-value).
	Q = defaultdict(lambda: np.zeros(env.action_space.n))

	# The policy we're following
	policy = make_epsilon_greedy_policy(Q, epsilon, env.action_space.n)

	for i_episode in range(num_episodes):
		done = False
		state = env.reset()
		episode = []

		while not done:
			action = policy(state, i_episode)
			next_state, reward, done, _ = env.step(action)
			episode_rewards[i_episode] += reward
			episode.append((state, action, reward))
			state = next_state

		for state, action, reward in episode:
			firstOccurence = next(i for i, x in enumerate(episode) if x[0] == state and x[1] == action)
			G = sum(x[2]*discount_factor**i for i, x in enumerate(episode[firstOccurence:]))
			returns_sum[(state, action)] += G
			returns_count[(state, action)] += 1
			Q[state][action] = returns_sum[(state, action)]/returns_count[(state, action)]

	return Q, policy


nbEpisodes = 5000
alpha=np.log(0.000001)/nbEpisodes
env = gym.make('FrozenLake-v0')
# env = Monitor(env, '/tmp/FrozenLake-v0', force=True)
episode_rewards = np.zeros(nbEpisodes)

Q, policy = mc_control_epsilon_greedy(env, num_episodes=nbEpisodes, epsilon=0.1)
# env.close()
# gym.upload('/tmp/FrozenLake-v0', api_key='sk_QoYvL963TwnAqSJXZLOQ')# matplotlib.style.use('ggplot')
plt.plot(range(nbEpisodes), episode_rewards)
plt.ylabel('Reward by episode')
plt.show()
