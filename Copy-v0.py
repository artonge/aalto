#####
# Monte Carlo
#####
import gym
from gym.wrappers.monitoring import Monitor
from random import randint
import numpy as np
import matplotlib.pyplot as plt

def learn(episodeCount):
	for i_episode in xrange(episodeCount):
		state     = env.reset()
		actionStr = policy(state, i_episode)
		done = False
		while not done:
			nextState, reward, done, _ = env.step(A[actionStr])
			nextActionStr = policy(nextState, i_episode)
			stepsHistory[i_episode] += reward
			updatePolicy(state, actionStr, reward, nextState, nextActionStr)
			state     = nextState
			actionStr = nextActionStr


def updatePolicy(state, actionStr, reward, nextstate, nextactionStr):
	y = 0.3
	l = 0.6
	alpha = 0.9
	d = reward + y*Q[nextstate][nextactionStr] - Q[state][actionStr]
	Z[state][actionStr] += 1
	for s in Q:
		for aStr in Q[s]:
			Q[s][aStr] = Q[s][aStr] + alpha*d*Z[s][aStr]
			Z[s][aStr] = y*l*Z[s][aStr]


def policy(state, i_episode):
	if state not in Q: Q[state] = {}
	if state not in Z: Z[state] = {}
	# Get the most valued action
	maxValueActionStr = None
	for actionStr in Q[state]:
		if maxValueActionStr == None or Q[state][actionStr] > Q[state][maxValueActionStr]:
			maxValueActionStr = actionStr
	# If no maxValueAction or if it's time for exploration, return random action
	if maxValueActionStr == None or randint(0, 100) < 20*0.99**i_episode:
		randomAction    = env.action_space.sample()
		randomActionStr = str(randomAction[0]) + str(randomAction[1]) + str(randomAction[2])
		if randomActionStr not in A       : A       [randomActionStr] = randomAction
		if randomActionStr not in Q[state]: Q[state][randomActionStr] = 0
		if randomActionStr not in Z[state]: Z[state][randomActionStr] = 0		
		return randomActionStr
	else:
		return maxValueActionStr


nbEpisodes = 10000

env = gym.make('Copy-v0')
# env = Monitor(env, 'tmp/copy', force=True)

stepsHistory = [0]*nbEpisodes

Q = {}  # state-action values
Z = {}  # state-action eligibility traces
A = {}  # actionStr to action map

learn(nbEpisodes)

env.close()
plt.plot(range(nbEpisodes), stepsHistory, range(nbEpisodes), [25]*nbEpisodes)
plt.ylabel('Number of steps')
plt.show()
