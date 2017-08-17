import gym
from gym.wrappers.monitoring import Monitor
from random import randint
import math
import matplotlib.pyplot as plt
import numpy as np


def learn(episodeCount):
	for i_episode in xrange(episodeCount):
		done = False
		obs = env.reset()
		nextstate = getState(obs)
		nextaction = policy(nextstate, i_episode)
		while not done:
			# TAKE STEP
			state  = nextstate
			action = nextaction
			obs, reward, done, _ = env.step(action)
			nextstate = getState(obs)
			nextaction = policy(nextstate, i_episode)

			# UPDATE Q
			discount = 0.99
			alpha    = 0.75
			d = reward + discount * Q[nextstate][np.argmax(Q[nextstate])] - Q[state][action]
			Q[state][action] = alpha * d

			# UPDATE STATS
			rewardHistory[i_episode] += reward


def getState(obs):
	state = ''
	for o in obs:
		state += str(int(math.floor(o*10)))+":"
	return state[:-1]


def policy(state, i_episode):
	if state not in Q:
		Q[state] = [0]*env.action_space.n
	if randint(0, 100) < 10*0.99**i_episode:
		explorationHistory[i_episode] += 1
		return env.action_space.sample()
	else:
		return np.argmax(Q[state])


nbEpisodes = 1000
rewardHistory      = [200]*nbEpisodes
explorationHistory = [0]*nbEpisodes

env = gym.make('MountainCar-v0')
# env = Monitor(env, 'tmp/cart-pole', force=True)

Q = {}
learn(nbEpisodes)
env.close()
print Q
plt.plot(range(nbEpisodes), rewardHistory, range(nbEpisodes), explorationHistory, range(nbEpisodes), [-110+200]*nbEpisodes)
plt.ylabel('Number of rewards')
plt.show()
