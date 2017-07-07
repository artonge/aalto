import gym
# from gym.wrappers.monitoring import Monitor
from random import randint
import time
import math
import matplotlib.pyplot as plt
import numpy as np


def learn(episodeCount, stepsHistory, decayHistory, i_learn):
	for i_episode in range(episodeCount):  # Start an episode
		obs = env.reset()
		# Reset reward count and states/actions for this episode
		episodeStatesActions = []
		totalRewards = 0
		# Compute the decay of the exploration
		decayX = 0.1
		decayY = 40
		decay = max(-i_episode*decayX+decayY, 10/(i_episode+1))
		decayHistory[i_episode] = decay

		for t in range(200):
			state = getState(obs)  # Get the state
			action = policy(state, decay)  # Get the action
			episodeStatesActions.append({'state': state, 'action': action})  # Save state and action to episodeStatesActions
			obs, reward, done, _ = env.step(action)  # Apply the action
			totalRewards += reward  # Update total reward for this episode
			if done:  # Episode is over
				stepsHistory[i_episode] = (stepsHistory[i_episode]*i_learn + t+1)/(i_learn+1)
				for state_action in episodeStatesActions:  # Update value for chosen actions
					updatePolicy(state_action['state'], state_action['action'], totalRewards)
				break


# @param state <string> the state to update
# @param action <int> the action to update
# @param <int> the reward
def updatePolicy(state, action, rewards):
	a = history[state][action]
	a['value'] = (a['value'] * a['count'] + rewards) / (a['count'] + 1)
	a['count'] += 1


# @param obs <[float]> the observation to convert into a state
# @return the state associated to the observation_space
# If the set of observations where never met, create the state
# The function reduces the number of possible states
def getState(obs):
	return str(math.floor(obs[0]))
	+str(math.floor(obs[1]))
	+str(math.floor(obs[2]))
	+str(math.floor(obs[3]))


# @param state <string>
# @param episode <int>
# @return an action
# The policy progressivly stops exploration and get greedy
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
	if randint(0, 100) < decay:
		return minCountAction
	else:
		return maxValueAction


nbEpisodes = 1000
stepsHistory = [0]*nbEpisodes
env = gym.make('CartPole-v0')
for i in range(3):
	print i
	history = {}  # 'state' ==> 'action'
	decayHistory = [0]*nbEpisodes
	learn(nbEpisodes, stepsHistory, decayHistory, i)
env.close()

plt.plot(range(nbEpisodes), stepsHistory, range(nbEpisodes), decayHistory)
plt.ylabel('Number of steps')
plt.show()
