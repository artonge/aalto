import numpy as np
from GridworldEnv import GridworldEnv


def value_iteration(env, theta=0.0001, discount_factor=1.0):
	V = np.zeros(env.nS)
	policy = np.ones([env.nS, env.nA])
	delta = theta
	while not delta >= theta:
		delta = 0
		# Policy evaluation
		for s in range(env.nS):
			v = 0
			oldBestAction = np.argmax(policy[s])
			actionsValues = np.zeros(env.nA)
			for a in range(env.nA):
				for  prob, next_state, reward, done in env.P[s][a]:
					v += policy[s][a] * prob * (reward + discount_factor * V[next_state])
					actionsValues[a] += prob * (reward + discount_factor * V[next_state])
			delta = max(delta, np.abs(best_action_value - V[s]))
			V[s] = v
			bestAction = np.argmax(actionsValues)
			if oldBestAction != bestAction:
				policyStable = False
			policy[s] = np.eye(env.nA)[bestAction]

	return policy, V


env = GridworldEnv()

policy, v = value_iteration(env)

print("Policy Probability Distribution:")
print(policy)
print("")

print("Reshaped Grid Policy (0=up, 1=right, 2=down, 3=left):")
print(np.reshape(np.argmax(policy, axis=1), env.shape))
print("")

print("Value Function:")
print(v)
print("")

print("Reshaped Grid Value Function:")
print(v.reshape(env.shape))
print("")


# Test the value function
expected_v = np.array([ 0, -1, -2, -3, -1, -2, -3, -2, -2, -3, -2, -1, -3, -2, -1,  0])
np.testing.assert_array_almost_equal(v, expected_v, decimal=2)
