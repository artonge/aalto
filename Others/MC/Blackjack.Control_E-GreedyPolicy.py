import gym
import matplotlib
import numpy as np
from collections import defaultdict
from BlackjackEnv import BlackjackEnv
import plotting


import sys
if "../" not in sys.path:
	sys.path.append("../")



def make_epsilon_greedy_policy(Q, epsilon, nA):
	def policy_fn(observation):
		A = np.ones(env.nA) * epsilon / nA
		A[np.argmax(Q[observation])] += 1 - epsilon
		return A

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
			probs = policy(state)
			action = np.random.choice(np.arange(len(probs)), p=probs)
			next_state, reward, done, _ = env.step(action)
			episode.append((state, action, reward))
			state = next_state

		for state, action, reward in episode:
			firstOccurence = next(i for i, x in enumerate(episode) if x[0] == state and x[1] == action)
			G = sum(x[2]*discount_factor**i for i, x in enumerate(episode[firstOccurence:]))
			returns_sum[(state, action)] += G
			returns_count[(state, action)] += 1
			Q[state][action] = returns_sum[(state, action)]/returns_count[(state, action)]

	return Q, policy


matplotlib.style.use('ggplot')
env = BlackjackEnv()
Q, policy = mc_control_epsilon_greedy(env, num_episodes=500000, epsilon=0.1)


# For plotting: Create value function from action-value function
# by picking the best action at each state
V = defaultdict(float)
for state, actions in Q.items():
	action_value = np.max(actions)
	V[state] = action_value
plotting.plot_value_function(V, title="Optimal Value Function")
