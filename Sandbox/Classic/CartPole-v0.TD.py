import gym
from random import randint
import math
import matplotlib.pyplot as plt
import numpy as np


def learn(episodeCount):

	for i_episode in range(episodeCount):  # Start an episode

		obs = env.reset()
		nextstate = getState(obs)

		for t in range(200):
			# TAKE STEP
			state = nextstate
			action = policy(state, i_episode)
			obs, reward, done, _ = env.step(action)
			nextstate = getState(obs)

			# UPDATE Q
			discount = 0.5
			alpha = 0.5
			d = reward + discount * Q[nextstate][np.argmax(Q[nextstate])] - Q[state][action]
			Q[state][action] = alpha * d

			# UPDATE STATS
			stepsHistory[i_episode] = t

			if done:  # Episode is over
				break



# @param obs <[float]> the observation to convert into a state
# @return the state associated to the observation_space
# If the set of observations where never met, create the state
# The function reduces the number of possible states
def getState(obs):
	state = ''
	for o in obs:
		state += str(math.floor(o))
	if state not in Q:  # If state does not existe in Q, create it
		Q[state] = [0]*env.action_space.n
	return state



# @param state <string>
# @param decay <int>
# @return an action
# The policy progressivly stops exploration and gets greedy
def policy(state, i_episode):
	#decay = 50*0.995**i_episode
	if randint(0, 100) < 10:
		exploration[i_episode] += 1
		return env.action_space.sample()
	else:
		return np.argmax(Q[state])


env = gym.make('CartPole-v0')
episodeCount = 1000
stepsHistory = [0]*episodeCount
exploration = [0]*episodeCount

Q = {}  # 'state' ==> [value <int>]

learn(episodeCount)

for s in Q:
	print s, Q[s]

env.close()

plt.plot(range(episodeCount), stepsHistory, range(episodeCount), exploration, range(episodeCount), [195]*episodeCount)
plt.ylabel('Number of steps')
plt.show()
