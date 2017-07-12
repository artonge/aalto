import gym
# from gym.wrappers.monitoring import Monitor
from random import randint
import math
import matplotlib.pyplot as plt
import numpy as np

def learn(episodeCount, stepsHistory, decayHistory, i_learn):
	for i_episode in range(episodeCount):  # Start an episode
		if i_episode % 1000 == 0: print i_episode
		# Compute the decay of the exploration
		decayX = 0.1
		decayY = 99
		decay = max(-i_episode*decayX+decayY, 10/(i_episode+1))
		#doEpisodeMC(decay, i_episode, i_learn)
		doEpisodeTD(decay, i_episode, i_learn)


def doEpisodeMC(decay, i_episode, i_learn):
	obs = env.reset()
	episodeStatesActions = []
	totalRewards = 0
	for t in range(200):
		state = getState(obs)  # Get the state
		action = policy(state, decay)  # Get the action
		episodeStatesActions.append({'state': state, 'action': action})  # Save state and action to episodeStatesActions
		obs, reward, done, _ = env.step(action)  # Apply the action
		totalRewards += reward  # Update total reward for this episode
		if done:  # Episode is over
			stepsHistory[i_episode] = totalRewards
			for i, state_action in enumerate(episodeStatesActions):  # Update value for chosen actions
				updatePolicyMCa(state_action['state'], state_action['action'], totalRewards-i)
			break


def doEpisodeTD(decay, i_episode, i_learn):
	obs = env.reset()
	episodeStatesActions = []
	lastState  = None
	lastAction = None
	reward     = 0
	totalRewards = 0
	for t in range(200):
		state = getState(obs)  # Get the state
		action = policy(state, i_episode)  # Get the action
		episodeStatesActions.append({'state': state, 'action': action})  # Save state and action to episodeStatesActions
		if t > 0:
			updatePolicyTD(lastState, lastAction, state, action, reward)
		obs, reward, done, _ = env.step(action)  # Apply the action
		totalRewards += reward
		lastState  = state
		lastAction = action
		if done:  # Episode is over
			stepsHistory[i_episode] = totalRewards
			break


# @param state <string> the state to update
# @param action <int> the action to update
# @param G <int> the reward
def updatePolicyMC(state, action, G):
	a = history[state][action]
	a['value'] = (a['value'] * a['count'] + G) / (a['count'] + 1)
	a['count'] += 1


def updatePolicyMCa(state, action, G):
	a = history[state][action]
	a['value'] = a['value'] + 0.1 * (G - a['value'])


# @param state <string> the state to update
# @param action <int> the action to update
# @param <int> the reward
def updatePolicyTD(state1, action1, state2, action2, reward):
	a1 = history[state1][action1]
	a2 = history[state2][action2]
	a1['value'] = a1['value'] + 0.5 * (reward + 0.95 * a2['value'] - a1['value'])
	a1['count'] += 1


# @param obs <[float]> the observation to convert into a state
# @return the state associated to the observation_space
# If the set of observations where never met, create the state
# The function slice the observation's space into nbStates
# This reduce the number of possible states
def getState(obs):
	state = ''
	for o in obs:
		state += str(int(math.floor(o*10)))+":"
	return state[:-1]


# @param state <string>
# @param decay <int>
# @return an action
# The policy progressivly stops exploration and gets greedy
def policy(state, i_episode):
	if state not in history:  # If state does not existe, create it
		history[state] = []
		for _ in range(env.action_space.n):
			history[state].append({'count': 0, 'value': 0})
	stateValues = history[state]
	# Get the less explored action and the most valued action
	maxValueAction = env.action_space.sample()
	minCountAction = env.action_space.sample()
	for action in range(env.action_space.n):
		if stateValues[maxValueAction]['value'] < stateValues[action]['value']:
			maxValueAction = action
		if stateValues[minCountAction]['count'] > stateValues[action]['count']:
			minCountAction = action
	# Computing the decay of the exploration
	if randint(0, 100) < 10 and stateValues[minCountAction]['count'] < 100:
		decayHistory[i_episode] -= 1
		return minCountAction
	else:
		return maxValueAction



nbEpisodes = 10000
stepsHistory = [0]*nbEpisodes
env = gym.make('MountainCar-v0')
# env = Monitor(env, 'tmp/cart-pole', force=True)
for i in range(1):
	print i
	history = {}  # 'state' ==> [{'count': int, 'value': float}]
	decayHistory = [-100]*nbEpisodes
	learn(nbEpisodes, stepsHistory, decayHistory, i)
env.close()
for key in history.keys():
	totalCount = 0
	for v in history[key]:
		totalCount += v["count"]
	print key, "	", totalCount
# gym.upload('tmp/cart-pole', api_key='sk_QoYvL963TwnAqSJXZLOQ')
plt.plot(range(nbEpisodes), stepsHistory, range(nbEpisodes), decayHistory, range(nbEpisodes), [-110]*nbEpisodes)
plt.ylabel('Number of steps')
plt.show()
