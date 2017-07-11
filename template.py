import gym
from gym.wrappers.monitoring import Monitor
from random import randint
import time
import math
import matplotlib.pyplot as plt
import numpy as np


def learn(episodeCount, stepsHistory, decayHistory, i_learn):
	for i_episode in range(episodeCount):  # Start an episode
		obs = env.reset()
		# Reset reward count and states/actions for this episode
		# Compute the decay of the exploration
		decayX = 0.5
		decayY = 50
		decay = max(-i_episode*decayX+decayY, 10/(i_episode+1))
		decayHistory[i_episode] = decay
		#doEpisodeMC(obs, decay, i_episode, i_learn)
		doEpisodeTD(obs, decay, i_episode, i_learn)


def doEpisodeMC(obs, decay, i_episode, i_learn):
	episodeStatesActions = []
	totalRewards = 0
	for t in range(200):
		state = getState(obs)  # Get the state
		action = policy(state, decay)  # Get the action
		episodeStatesActions.append({'state': state, 'action': action})  # Save state and action to episodeStatesActions
		obs, reward, done, _ = env.step(action)  # Apply the action
		totalRewards += reward  # Update total reward for this episode
		if done:  # Episode is over
			stepsHistory[i_episode] = (stepsHistory[i_episode]*i_learn + t+1)/(i_learn+1)
			for i, state_action in enumerate(episodeStatesActions):  # Update value for chosen actions
				updatePolicyMCa(state_action['state'], state_action['action'], totalReward-i)
			break


def doEpisodeTD(obs, decay, i_episode, i_learn):
	episodeStatesActions = []
	lastState  = None
	lastAction = None
	for t in range(200):
		state = getState(obs)  # Get the state
		action = policy(state, decay)  # Get the action
		episodeStatesActions.append({'state': state, 'action': action})  # Save state and action to episodeStatesActions
		if t > 0:
			updatePolicyTD(lastState, lastAction, state, action, reward)
		obs, reward, done, _ = env.step(action)  # Apply the action
		lastState  = state
		lastAction = action
		if done:  # Episode is over
			stepsHistory[i_episode] = (stepsHistory[i_episode]*i_learn + t+1)/(i_learn+1)
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
	a1['value'] = a1['value'] + 0.1 * (reward + 1 * a2['value'] - a1['value'])
	a1['count'] += 1


# @param obs <[float]> the observation to convert into a state
# @return the state associated to the observation_space
# If the set of observations where never met, create the state
# The function reduces the number of possible states
def getState(obs):
	state = ''
	state += str(math.floor(obs[0]))
	state += str(math.floor(obs[1]))
	state += str(math.floor(obs[2]))
	state += str(math.floor(obs[3]))
	return state


# @param state <string>
# @param decay <int>
# @return an action
# The policy progressivly stops exploration and gets greedy
def policy(state, decay):
	# Get the less explored action and the most valued action
	maxValueAction = env.action_space.sample()
	minCountAction = env.action_space.sample()
	if not history.has_key(state):  # If state does not existe, create it
		history[state] = []
		for _ in range(env.action_space.n):
			history[state].append({'count':0, 'value':0})
	stateValues = history[state]
	for action in range(env.action_space.n):
		if stateValues[maxValueAction]['value'] < stateValues[action]['value']:
			maxValueAction = action
		if stateValues[minCountAction]['count'] > stateValues[action]['count']:
			minCountAction = action
	# Computing the decay of the exploration
	if randint(0, 100) < 10:
		return minCountAction
	else:
		return maxValueAction


nbEpisodes = 5000
stepsHistory = [0]*nbEpisodes
env = gym.make('CartPole-v0')
#env = Monitor(env, 'tmp/cart-pole', force=True)
for i in range(6):
	print i
	history = {}  # 'state' ==> [{'count': int, 'value': float}]
	decayHistory = [0]*nbEpisodes
	learn(nbEpisodes, stepsHistory, decayHistory, i)
env.close()
#gym.upload('tmp/cart-pole', api_key='sk_QoYvL963TwnAqSJXZLOQ') 
plt.plot(range(nbEpisodes), stepsHistory, range(nbEpisodes), decayHistory, range(nbEpisodes), [195]*nbEpisodes)
plt.ylabel('Number of steps')
plt.show()
