import gym
from gym.wrappers.monitoring import Monitor
from random import randint
import time
import math
import matplotlib.pyplot as plt
import numpy as np


def learn(episodeCount, stepsHistory, decayHistory):
	for i_episode in range(episodeCount):  # Start an episode
		obs = env.reset()
		# Reset reward count and states/actions for this episode
		# Compute the decay of the exploration
		decayX = 0.5
		decayY = 50
		decay = max(-i_episode*decayX+decayY, 10/(i_episode+1))
		decayHistory[i_episode] = decay
		#doEpisodeMC(obs, decay, i_episode)
		doEpisodeTD(obs, decay, i_episode)


def doEpisodeMC(obs, decay, i_episode):
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


def doEpisodeTD(obs, decay, i_episode):
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
		stepsHistory[i_episode] += reward
		if done:  # Episode is over
			break


# @param state <string> the state to update
# @param action <int> the action to update
# @param G <int> the reward
def updatePolicyMC(state, action, G):
	a = history[state][stringifyBox(action)]
	a['value'] = (a['value'] * a['count'] + G) / (a['count'] + 1)
	a['count'] += 1


def updatePolicyMCa(state, action, G):
	a = history[state][stringifyBox(action)]
	a['value'] = a['value'] + 0.1 * (G - a['value'])


# @param state <string> the state to update
# @param action <int> the action to update
# @param <int> the reward
def updatePolicyTD(state1, action1, state2, action2, reward):
	a1 = history[state1][stringifyBox(action1)]
	a2 = history[state2][stringifyBox(action2)]
	a1['value'] = a1['value'] + 0.05 * (reward + 1 * a2['value'] - a1['value'])
	a1['count'] += 1


# @param obs <[float]> the observation to convert into a state
# @return the state associated to the observation_space
# If the set of observations where never met, create the state
# The function reduces the number of possible states
def getState(obs):
	return stringifyBox(obs*1)

def stringifyBox(box):
	s = ''
	for b in box:
		s += str(math.floor(b))
	return s

# @param state <string>
# @param decay <int>
# @return an action
# The policy progressivly stops exploration and gets greedy
def policy(state, decay):
	if not history.has_key(state):  # If state does not existe, create it
		history[state] = {}
	stateValues = history[state]
	if randint(0, 100) < 10 or len(stateValues.keys()) == 0:
		action = env.action_space.sample()
		stateValues[stringifyBox(action)] = {'value': 0, 'count': 0, 'action': action}
		return action
	else:
		# Get the less explored action and the most valued action
		maxValueAction = None		
		for action in stateValues.keys():
			if maxValueAction == None or stateValues[stringifyBox(maxValueAction)]['value'] < stateValues[action]['value']:
				maxValueAction = stateValues[action]['action']
		
		return maxValueAction


nbEpisodes = 2000
stepsHistory = [0]*nbEpisodes
env = gym.make('Pendulum-v0')
#env = Monitor(env, 'tmp/cart-pole', force=True)
history = {}  # 'state' ==> [{'count': int, 'value': float}]
decayHistory = [0]*nbEpisodes

learn(nbEpisodes, stepsHistory, decayHistory)
env.close()
for state in history.keys():
	print state, history[state]
#gym.upload('tmp/pendulum', api_key='sk_QoYvL963TwnAqSJXZLOQ') 
plt.plot(range(nbEpisodes), stepsHistory, range(nbEpisodes), decayHistory, range(nbEpisodes), [195]*nbEpisodes)
plt.ylabel('Number of steps')
plt.show()
