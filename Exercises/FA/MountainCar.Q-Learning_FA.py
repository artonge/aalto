%matplotlib inline

import gym
import matplotlib
import numpy as np
import sys
import sklearn.pipeline
import sklearn.preprocessing
from sklearn.linear_model import SGDRegressor
from sklearn.kernel_approximation import RBFSampler
import plotting


class Estimator():
	def __init__(self):
		# We create a separate model for each action in the environment's
		# action space. Alternatively we could somehow encode the action
		# into the features, but this way it's easier to code up.
		self.models = []
		for _ in range(env.action_space.n):
			model = SGDRegressor(learning_rate="constant")
			# We need to call partial_fit once to initialize the model
			# or we get a NotFittedError when trying to make a prediction
			# This is quite hacky.
			model.partial_fit([self.featurize_state(env.reset())], [0])
			self.models.append(model)

	def featurize_state(self, state):
		scaled = scaler.transform([state])
		featurized = featurizer.transform(scaled)
		return featurized[0]

	def predict(self, s, a=None):
		features = self.featurize_state(s)
		if a:
			return self.models[a].predict([features])[0]
		else:
			return np.array([m.predict([features])[0] for m in self.models])

	def update(self, s, a, y):
		features = self.featurize_state(s)
		self.models[a].partial_fit([features], [y])
		return None


def make_epsilon_greedy_policy(estimator, epsilon, nA):
	def policy_fn(observation):
		A = np.ones(nA, dtype=float) * epsilon / nA
		q_values = estimator.predict(observation)
		best_action = np.argmax(q_values)
		A[best_action] += (1.0 - epsilon)
		return A
	return policy_fn



def q_learning(env, estimator, num_episodes, discount_factor=1.0, epsilon=0.1, epsilon_decay=1.0):
	# Keeps track of useful statistics
	stats = plotting.EpisodeStats(
		episode_lengths=np.zeros(num_episodes),
		episode_rewards=np.zeros(num_episodes)
	)

	for i_episode in range(num_episodes):

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
			probs = policy(state)
			action = np.random.choice(np.arange(len(probs)), p=probs)
			nextstate, reward, done, _ = env.step(action)

			target = reward + discount_factor * np.max(estimator.predict(nextstate))
			estimator.update(state, action, target)

			stats.episode_lengths[i_episode] += 1
			stats.episode_rewards[i_episode] += reward

			state = nextstate

	return stats



matplotlib.style.use('ggplot')

env = gym.envs.make("MountainCar-v0")


# Feature Preprocessing: Normalize to zero mean and unit variance
# We use a few samples from the observation space to do this
observation_examples = np.array([env.observation_space.sample() for x in range(10000)])
scaler = sklearn.preprocessing.StandardScaler()
scaler.fit(observation_examples)

# Used to convert a state to a featurized representation.
# We use RBF kernels with different variances to cover different parts of the space
featurizer = sklearn.pipeline.FeatureUnion([
		("rbf1", RBFSampler(gamma=5.0, n_components=100)),
		("rbf2", RBFSampler(gamma=2.0, n_components=100)),
		("rbf3", RBFSampler(gamma=1.0, n_components=100)),
		("rbf4", RBFSampler(gamma=0.5, n_components=100))
])
featurizer.fit(scaler.transform(observation_examples))


estimator = Estimator()

# Note: For the Mountain Car we don't actually need an epsilon > 0.0
# because our initial estimate for all states is too "optimistic" which leads
# to the exploration of all states.
stats = q_learning(env, estimator, 100, epsilon=0.0)

plotting.plot_cost_to_go_mountain_car(env, estimator)
plotting.plot_episode_stats(stats, smoothing_window=25)
