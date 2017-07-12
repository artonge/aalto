import gym
from gym.wrappers.monitoring import Monitor
from random import randint
import math
import matplotlib.pyplot as plt


def learn(episodeCount, i_learn):
	for i_episode in range(episodeCount):  # Start an episode
		#doEpisodeMC(i_episode, i_learn)
		doEpisodeTD(i_episode, i_learn)


def doEpisodeMC(i_episode, i_learn):
	obs = env.reset()
	episodeStatesActions = []
	totalRewards = 0
	for t in range(200):
		state = getState(obs)  # Get the state
		action = policy(state, i_episode)  # Get the action
		episodeStatesActions.append({'state': state, 'action': action})  # Save state and action to episodeStatesActions
		obs, reward, done, _ = env.step(action)  # Apply the action
		totalRewards += reward  # Update total reward for this episode
		if done:  # Episode is over
			stepsHistory[i_episode] = (stepsHistory[i_episode]*i_learn + t+1)/(i_learn+1)
			for i, state_action in enumerate(episodeStatesActions):  # Update value for chosen actions
				updatePolicyMC(state_action['state'], state_action['action'], totalRewards-i)
			break


def doEpisodeTD(i_episode, i_learn):
	obs = env.reset()
	episodeStatesActions = []
	lastState  = None
	lastAction = None
	reward     = 0
	for t in range(2000):
		state = getState(obs)  # Get the state
		action = policy(state, i_episode)  # Get the action
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
	for o in obs:
		state += str(math.floor(o))
	return state


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
	# Compute the decay of the exploration
	decayX = 0.5
	decayY = 50
	decay = max(-i_episode*decayX+decayY, 10/(i_episode+1))
	if randint(0, 100) < decay:
		explorationHistory[i_episode] += 1
		return minCountAction
	else:
		return maxValueAction


nbEpisodes = 1000
stepsHistory = [0]*nbEpisodes
env = gym.make('LunarLander-v2')
env = Monitor(env, 'tmp/cart-pole', force=True)
for i in range(6):
	print i
	history = {}  # 'state' ==> [{'count': int, 'value': float}]
	explorationHistory = [0]*nbEpisodes
	learn(nbEpisodes, i)
env.close()
# gym.upload('tmp/cart-pole', api_key='sk_QoYvL963TwnAqSJXZLOQ')
plt.plot(range(nbEpisodes), stepsHistory, range(nbEpisodes), explorationHistory, range(nbEpisodes), [195]*nbEpisodes)
plt.ylabel('Number of rewards')
plt.show()
