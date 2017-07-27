import numpy as np
from GridworldEnv import GridworldEnv


def policy_eval(policy, env, discount_factor=1.0, theta=0.00001):
	# Start with a random (all 0) value function
	V = np.zeros(env.nS)
	while True:
		delta = 0
		# For each state, perform a "full backup"
		for s in range(env.nS):
			v = 0
			# Look at the possible next actions
			for a, action_prob in enumerate(policy[s]):
				# For each action, look at the possible next states...
				for  prob, next_state, reward, done in env.P[s][a]:
					# Calculate the expected value
					v += action_prob * prob * (reward + discount_factor * V[next_state])
			# How much our value function changed (across any states)
			delta = max(delta, np.abs(v - V[s]))
			V[s] = v
		# Stop evaluating once our value function change is below a threshold
		if delta < theta:
			break
	return np.array(V)


def policy_improvement(env, policy_eval_fn=policy_eval, discount_factor=1.0):
	# Start with a random policy
	policy = np.ones([env.nS, env.nA]) / env.nA
	policyStable = False

	while not policyStable:
		V = policy_eval(policy, env, discount_factor)
		policyStable = True

		for s in range(env.nS):
			oldBestAction = np.argmax(policy[s])
			actionsValues = np.zeros(env.nA)
			for a in range(env.nA):
				for prob, next_state, reward, done in env.P[s][a]:
					actionsValues[a] += prob * (reward + discount_factor * V[next_state])

			bestAction = np.argmax(actionsValues)
			if oldBestAction != bestAction:
				policyStable = False

			policy[s] = np.eye(env.nA)[bestAction]

	return policy, V


env = GridworldEnv()

policy, v = policy_improvement(env)

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
