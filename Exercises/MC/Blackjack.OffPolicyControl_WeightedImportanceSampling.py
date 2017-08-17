import gym
import matplotlib
import numpy as np
from collections import defaultdict
from BlackjackEnv import BlackjackEnv
import plotting
import sys

def create_random_policy(nA):
	A = np.ones(nA, dtype=float) / nA
	def policy_fn(observation):
		return A
	return policy_fn


def create_greedy_policy(Q):
	def policy_fn(observation):
		A = np.zeros_like(Q[observation], dtype=float)
		A[np.argmax(Q[observation])] = 1
		return A
	return policy_fn


def mc_control_importance_sampling(env, num_episodes, behavior_policy, discount_factor=1.0):
	"""
	Monte Carlo Control Off-Policy Control using Weighted Importance Sampling.
	Finds an optimal greedy policy.

	Args:
		env: OpenAI gym environment.
		num_episodes: Nubmer of episodes to sample.
		behavior_policy: The behavior to follow while generating episodes.
			A function that given an observation returns a vector of probabilities for each action.
		discount_factor: Lambda discount factor.

	Returns:
		A tuple (Q, policy).
		Q is a dictionary mapping state -> action values.
		policy is a function that takes an observation as an argument and returns
		action probabilities. This is the optimal greedy policy.
	"""

	# The final action-value function.
	# A dictionary that maps state -> action values
	Q = defaultdict(lambda: np.zeros(env.action_space.n))
	C = defaultdict(lambda: np.zeros(env.action_space.n))
	# Our greedily policy we want to learn
	target_policy = create_greedy_policy(Q)

	for i_episode in range(num_episodes):
		if i_episode % 1000 == 0:
			print("\rEpisode {}/{}.".format(i_episode, num_episodes), end="")
			sys.stdout.flush()

		done = False
		state = env.reset()
		episode = []

		while not done:
			probs = behavior_policy(state)
			action = np.random.choice(np.arange(len(probs)), p=probs)
			next_state, reward, done, _ = env.step(action)
			episode.append((state, action, reward))
			state = next_state

		G = 0.0
		W = 1.0
		for t in range(len(episode))[::-1]:
			state, action, reward = episode[t]
			G = discount_factor * G + reward

			C[state][action] += W
			Q[state][action] += (W / C[state][action]) * (G - Q[state][action])

			if action != np.argmax(target_policy(state)):
				break

			W *= 1./behavior_policy(state)[action]

	return Q, target_policy


matplotlib.style.use('ggplot')
env = BlackjackEnv()
random_policy = create_random_policy(env.action_space.n)
Q, policy = mc_control_importance_sampling(env, num_episodes=500000, behavior_policy=random_policy)



# For plotting: Create value function from action-value function
# by picking the best action at each state
V = defaultdict(float)
for state, action_values in Q.items():
	action_value = np.max(action_values)
	V[state] = action_value
plotting.plot_value_function(V, title="Optimal Value Function")
