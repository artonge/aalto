import gym
from random import randint
import math
import matplotlib.pyplot as plt
from gym.wrappers.monitoring import Monitor

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
			totalRewards += reward  # Update total reward for this episode
			if done:  # Episode is over
				stepsHistory[i_episode] = t
				for i, state_action in enumerate(episodeStatesActions):  # Update value for chosen actions
					updatePolicyMC(state_action['state'], state_action['action'], totalRewards-i)
				break


# @param state <string> the state to update
# @param action <int> the action to update
# @param G <int> the reward
def updatePolicyMC(state, action, G):
	a = history[state][action]
	a['value'] = (a['value'] * a['count'] + G) / (a['count'] + 1)
	a['count'] += 1


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
	# Get the less explored action and the most valued action
	maxA = env.action_space.sample()
	if state not in history:  # If state does not existe, create it
		history[state] = []
		for _ in range(env.action_space.n):
			history[state].append({'count': 0, 'value': 0})
	stateValues = history[state]
	for action in range(env.action_space.n):
		if stateValues[maxA]['value'] < stateValues[action]['value']:
			maxA = action
	# Computing the decay of the exploration
	decay = 15*0.99**i_episode
	decayHistory[i_episode] = decay
	if randint(0, 100) < decay:
		return env.action_space.sample()
	else:
		return maxA


env = gym.make('CartPole-v0')
env = Monitor(env, 'tmp/cart-pole', force=True)
nbEpisodes = 1000
stepsHistory = [0]*nbEpisodes
decayHistory = [0]*nbEpisodes

history = {}  # 'state' ==> [{'count': int, 'value': float}]

learn(nbEpisodes)

env.close()
plt.plot(range(nbEpisodes), stepsHistory, range(nbEpisodes), decayHistory, range(nbEpisodes), [195]*nbEpisodes)
plt.ylabel('Number of steps')
plt.show()
