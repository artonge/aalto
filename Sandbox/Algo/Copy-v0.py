import gym
from gym.wrappers.monitoring import Monitor
from random import randint
import numpy as np
import matplotlib.pyplot as plt

def learn(episodeCount):
	for i_episode in xrange(episodeCount):
		nextState     = env.reset()
		nextActionStr = policy(nextState, i_episode)
		done = False
		while not done:
			# TAKE STEP
			state     = nextState
			actionStr = nextActionStr
			nextState, reward, done, _ = env.step(A[actionStr])
			nextActionStr = policy(nextState, i_episode)
			stepsHistory[i_episode] += reward

			# UPDATE POLICY
			discount = 0.9
			alpha = 0.9
			d = reward + discount * Q[nextState][maxAction(Q[nextState])] - Q[state][actionStr]
			Q[state][actionStr] = alpha * d



def maxAction(stateActions):
	maxAStr = None
	for aStr in stateActions:
		if maxAStr == None or stateActions[aStr] > stateActions[maxAStr]:
			maxAStr = aStr
	if maxAStr == None:
		maxAStr = randomAction(stateActions)
	return maxAStr



def randomAction(stateActions):
	randomA    = env.action_space.sample()
	randomAStr = str(randomA[0]) + str(randomA[1]) + str(randomA[2])
	if randomAStr not in A:
		A[randomAStr] = randomA
	if randomAStr not in stateActions:
		stateActions[randomAStr] = 0
	return randomAStr



def policy(state, i_episode):
	if state not in Q: Q[state] = {}
	maxAStr = maxAction(Q[state])
	if randint(0, 100) < 50*0.9**i_episode:
		explorationHistory[i_episode] += 1
		return randomAction(Q[state])
	else:
		return maxAStr



nbEpisodes = 10000

env = gym.make('Copy-v0')
# env = Monitor(env, 'tmp/copy', force=True)

stepsHistory = [0]*nbEpisodes
explorationHistory = [0]*nbEpisodes

Q = {}  # state-action values
A = {}  # actionStr to action map

learn(nbEpisodes)

env.close()
plt.plot(range(nbEpisodes), stepsHistory, range(nbEpisodes), explorationHistory, range(nbEpisodes), [25]*nbEpisodes)
plt.ylabel('Number of steps')
plt.show()
