import gym
from random import randint
import math
import matplotlib.pyplot as plt
import numpy as np


def learn(episodeCount):
	for i_episode in range(episodeCount):  # Start an episode
		obs = env.reset()
		for s in V.keys():
			Z[s] = [0]*env.action_space.n
		for t in range(200):
			state = getState(obs)  # Get the state
			action = policy(state, i_episode)  # Get the action
			obs, reward, done, _ = env.step(action)  # Apply the action
			updateValue(state, action, reward, getState(obs))
			if done:  # Episode is over
				stepsHistory[i_episode] = t
				break


# @param state <string> the state to update
# @param action <int> the action to update
# @param G <int> the reward
def updateValue(state, action, reward, nextstate):
	y = 0.5
	l = 0.6
	a = 0.5
	d = reward + y*np.argmax(V[nextstate]) - V[state][action]
	Z[state][action] = 1
	for s in V.keys():
		for a, v in enumerate(V[s]):		
			V[s][a] = V[s][a] + a*d*Z[s][a]
			if a == 0: Zhistory[s].append(Z[s][a])
			Z[s][a] = y*l*Z[s][a]


# @param obs <[float]> the observation to convert into a state
# @return the state associated to the observation_space
# If the set of observations where never met, create the state
# The function reduces the number of possible states
def getState(obs):
	state = ''
	for o in obs:
		state += str(math.floor(o))
	if state not in V:  # If state does not existe in V, create it
		V[state] = [0]*env.action_space.n
		Z[state] = [0]*env.action_space.n
		Zhistory[state] = []
	return state


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
		return np.argmax(V[state])


env = gym.make('CartPole-v0')
episodeCount = 1000
stepsHistory = [0]*episodeCount
decayHistory = [0]*episodeCount

V = {}  # 'state' ==> [value <int>]
Z = {}  # 'state' ==> [value <int>]
Zhistory = {}

learn(episodeCount)

#for s in V.keys():
#	print s
#	print s, '	', Z[s], '	', V[s]

env.close()

plt.plot(range(episodeCount), stepsHistory, range(episodeCount), decayHistory, range(episodeCount), [195]*episodeCount)
#plt.plot(Zhistory[s])
plt.ylabel('Number of steps')
plt.show()

