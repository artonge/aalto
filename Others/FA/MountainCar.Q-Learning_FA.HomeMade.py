import gym
import matplotlib
import numpy as np
import sys
import plotting


def buildC(N, size, i=0, base=[]):
	if len(base) == 0: base = [0]*size
	if i == size     : return [base]
	C = np.empty([0, size], dtype=int)
	for n in range(N+1):
		b = np.array(base)
		b[i] = n
		C = np.concatenate((C, buildC(N, size, i+1, b)))
	return C


class Estimator():
	def __init__(self):
		featuresSample = self.featurize_state(env.observation_space.sample(), env.action_space.sample())
		self.w = np.zeros(len(featuresSample))

	def featurize_state(self, s, a):
		S = np.concatenate((s, [a]))
		N = 5
		d = len(S)
		C = buildC(N, d)
		X = np.ones(len(C))
		for i in range(len(C)):
			for j in range(len(S)):
				X[i] *= S[j] ** C[i][j]
		return X

	def predict(self, s, a=None):
		if a != None:
			# print(self,w, self.featurize_state(s, a), np.matmul(self.w, self.featurize_state(s, a)))
			return np.matmul(self.featurize_state(s, a).T, self.w)
		else:
			values = [0] * env.action_space.n
			for a in range(env.action_space.n):
				values[a] = np.matmul(self.w, self.featurize_state(s, a))
			return values

	def update(self, s, a, y):
		alpha = 0.01
		self.w += alpha * (y - self.predict(s, a)) * self.featurize_state(s, a)


def make_epsilon_greedy_policy(estimator, epsilon, nA):
	def policy_fn(state):
		A = np.ones(nA, dtype=float) * epsilon / nA
		values = estimator.predict(state)
		A[np.argmax(values)] += (1.0 - epsilon)
		return np.random.choice(np.arange(len(A)), p=A)
	return policy_fn


def q_learning(env, estimator, num_episodes, discount_factor=1.0, epsilon=0.1, epsilon_decay=1.0):
	# Keeps track of useful statistics
	stats = plotting.EpisodeStats(
		episode_lengths=np.zeros(num_episodes),
		episode_rewards=np.zeros(num_episodes)
	)

	for i_episode in range(num_episodes):
		# plotting.plot_cost_to_go_mountain_car(env, estimator)

		# The policy we're following
		policy = make_epsilon_greedy_policy(
			estimator,
			epsilon * epsilon_decay**i_episode,
			env.action_space.n
		)

		# Print out which episode we're on, useful for debugging.
		# Also print reward for last episode
		last_reward = stats.episode_rewards[i_episode - 1]
		print("\rEpisode {}/{} ({})".format(i_episode + 1, num_episodes, last_reward), end="")
		sys.stdout.flush()

		done = False
		state = env.reset()
		while not done:
			action = policy(state)
			nextstate, reward, done, _ = env.step(action)

			target = reward + discount_factor * np.max(estimator.predict(nextstate))
			estimator.update(state, action, target)

			stats.episode_lengths[i_episode] += 1
			stats.episode_rewards[i_episode] += reward

			state = nextstate

	return stats


matplotlib.style.use('ggplot')

env = gym.envs.make("MountainCar-v0")
estimator = Estimator()

stats = q_learning(env, estimator, 100, epsilon=0.0)

plotting.plot_cost_to_go_mountain_car(env, estimator)
# plotting.plot_episode_stats(stats, smoothing_window=25)
