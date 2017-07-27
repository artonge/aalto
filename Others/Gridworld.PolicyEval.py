# https://github.com/dennybritz/reinforcement-learning/tree/master/DP
import numpy as np
from GridworldEnv import GridworldEnv


def policy_eval(policy, env, discount_factor=1.0, theta=0.00001):
	V = np.zeros(env.nS)
	delta = theta
	while delta >= theta:
		delta = 0
		for s in range(env.nS):
			v = 0
			for a in range(env.nA):
					for prob, next_state, reward, done in env.P[s][a]:
						v += policy[s][a] * (reward + discount_factor * prob * V[next_state])
			delta = max(delta, np.abs(v - V[s]))
			V[s] = v
	return V


env = GridworldEnv()

random_policy = np.ones([env.nS, env.nA]) / env.nA
v = policy_eval(random_policy, env)

expected_v = np.array([0, -14, -20, -22, -14, -18, -20, -20, -20, -20, -18, -14, -22, -20, -14, 0])
np.testing.assert_array_almost_equal(v, expected_v, decimal=2)
