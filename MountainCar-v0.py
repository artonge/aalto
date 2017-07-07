import gym
# from gym.wrappers.monitoring import Monitor
from random import randint
import time
# import math
# import matplotlib.pyplot as plt
import numpy as np


def learn(episodeCount):
	for i_episode in range(episodeCount):  # Start an episode
		obs = env.reset()
		episodeStatesActions = []
		totalRewards = 0

		for t in range(200):
			state = getState(obs)  # Get the state
			action = policy(state, i_episode)  # Get the action
			episodeStatesActions.append({'state': state, 'action': action})  # Save state and action to episodeStatesActions
			obs, reward, done, _ = env.step(action)  # Apply the action
			if reward != -1:
				print i_episode, 'ok'
			totalRewards += reward  # Update total reward for this episode
			if done:  # Episode is over
				for state_action in episodeStatesActions:  # Update value for chosen actions
					updatePolicy(state_action['state'], state_action['action'], totalRewards)
				break


def updatePolicy(state, action, rewards):
		a = history[state][action]
		a['value'] = (a['value'] * a['count'] + rewards) / (a['count'] + 1)
		a['count'] += 1


# @param obs <[float]> the observation to convert into a state
# @return the state associated to the observation_space
# If the set of observations where never met, create the state
# The function slice the observation's space into nbStates
# This reduce the number of possible states
def getState(obs):
	nbStates = 4  # The number of slices
	low    = np.array(env.observation_space.low)
	high   = np.array(env.observation_space.high)
	delta  = (high-low)/(nbStates)  # The delta for each slices
	rawState = (np.array(obs)-low)/delta  # The raw state for the observations
	state = ''  # Compute the state
	for s in rawState: state += str(int(s))
	if not history.has_key(state):  # If state does not existe, create it
		history[state] = [{'count':0, 'value':0}]*env.action_space.n
	return state  # Return the state and the serialized state


# @param state <string>
# @param episode <int>
# @return an action
# The policy progressivly stops exploration and get greedy
def policy(state, episode):
	# Get the less explored action and the most valued action
	maxValueAction = env.action_space.sample()
	minCountAction = env.action_space.sample()
	stateValues = history[state]
	for action in range(env.action_space.n):
		if stateValues[maxValueAction]['value'] < stateValues[action]['value']:
			maxValueAction = action
		if stateValues[minCountAction]['count'] > stateValues[action]['count']:
			minCountAction = action
	# Computing the decay of the exploration
	decayX = 0.3
	decayY = 99
	decay = -episode*decayX+decayY
	if randint(0, 100) < decay:
		return minCountAction
	else:
		return maxValueAction


history = {}  # 'state' ==> 'action'
env = gym.make('MountainCar-v0')
learn(200)
print history.keys()
env.close()
