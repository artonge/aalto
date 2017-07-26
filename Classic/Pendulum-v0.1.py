import gym
from random import randint
import math
import matplotlib.pyplot as plt
import numpy as np


def learn(episodeCount):
	for i_episode in range(episodeCount):  # Start an episode
		obs = env.reset()
		for s in V.keys():
			Z[s] = {}
		totalRewards = 0
		for t in range(200):
			state = getState(obs)  # Get the state
			action = policy(state, i_episode)  # Get the action
			obs, reward, done, _ = env.step(action)  # Apply the action
			totalRewards += reward			
			updateValue(state, action, reward, getState(obs))
			if done:  # Episode is over
				stepsHistory[i_episode] = totalRewards
				break


def maxAction(stateActions):
	maxAction = None
	for a, v in stateActions.items():
		if maxAction == None or stateActions[maxAction]['value'] < v['value']:
			maxAction = a
	if maxAction == None:
		return {'action': env.action_space.sample(), 'value': 0}
	else:
		return stateActions[maxAction]


# @param state <string> the state to update
# @param action <int> the action to update
# @param G <int> the reward
def updateValue(state, action, reward, nextstate):
	y = 0.95
	l = 0.6
	alpha = 0.5
	if stringifyBox(action) not in V[state]:
		V[state][stringifyBox(action)] = {'action': action, 'value': 0}
	d = reward + y*maxAction(V[nextstate])['value'] - V[state][stringifyBox(action)]['value']
	Z[state][stringifyBox(action)] = 1
	for s in V:
		for a in V[s]:
			if a not in Z[s]:
				Z[s][a] = 0
			V[s][a]['value'] = V[s][a]['value'] + alpha*d*Z[s][a]
			Zhistory[s].append(Z[s][a])
			Z[s][a] = y*l*Z[s][a]


# @param obs <[float]> the observation to convert into a state
# @return the state associated to the observation_space
# If the set of observations where never met, create the state
# The function reduces the number of possible states
def getState(obs):
	state = stringifyBox(obs)
	if state not in V:  # If state does not existe in V, create it
		V[state] = {}
		Z[state] = {}
		Zhistory[state] = []
	return state


def stringifyBox(box):
	s = ''
	for b in box:
		s += str(math.floor(b))
	return s


# @param state <string>
# @param decay <int>
# @return an action
# The policy progressivly stops exploration and gets greedy
def policy(state, i_episode):
	decay = 50*0.95**i_episode
	decayHistory[i_episode] = decay
	if randint(0, 100) < decay:
		return env.action_space.sample()
	else:
		return maxAction(V[state])['action']


env = gym.make('Pendulum-v0')
episodeCount = 2000
stepsHistory = [0]*episodeCount
decayHistory = [0]*episodeCount


V = {}  # 'state' ==> [value <int>]
Z = {}  # 'state' ==> [value <int>]
actionMap = {} # 'stringifyed action' ==> raw action
Zhistory = {}

learn(episodeCount)

#for s in V.keys():
#	print s
#	print s, '	', Z[s], '	', V[s]

env.close()

plt.plot(range(episodeCount), stepsHistory, range(episodeCount), decayHistory, range(episodeCount), [-500]*episodeCount)
#plt.plot(Zhistory[s])
plt.ylabel('Number of steps')
plt.show()

