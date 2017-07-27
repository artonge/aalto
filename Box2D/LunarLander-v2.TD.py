import gym
from gym.wrappers.monitoring import Monitor
from random import randint
import math
import matplotlib.pyplot as plt
import numpy as np

def learn(episodeCount):
	for i_episode in range(episodeCount):
		done = False
		obs = env.reset()
		next_state  = getState(obs)
		next_action = policy(next_state, i_episode)
		if i_episode % 100 == 0: print i_episode
		while not done:
			# TAKE STEP
			state  = next_state
			action = next_action
			obs, reward, done, _ = env.step(action)
			next_state  = getState(obs)
			next_action = policy(next_state, i_episode)
			rewardHistory[i_episode] += reward

			# UPDATE Q
			discount = 0.7
			alpha = 0.5
			d = reward + discount * Q[next_state][np.argmax(Q[next_state])] - Q[state][action]
			Q[state][action] = alpha * d


def getState(obs):
	state = ''
	for o in obs:
		state += str(math.floor(o*10))
	return state


def policy(state, i_episode):
	# If state does not existe, create it
	if state not in Q:
		Q[state] = [0]*env.action_space.n
	if randint(0, 100) < 10:
		explorationHistory[i_episode] += 1
		return env.action_space.sample()
	else:
		return np.argmax(Q[state])


nbEpisodes = 1000

explorationHistory = [0]*nbEpisodes
rewardHistory      = [0]*nbEpisodes

env = gym.make('LunarLander-v2')
# env = Monitor(env, 'tmp/cart-pole', force=True)
print env.action_space
print env.observation_space

Q = {}

learn(nbEpisodes)
env.close()

# for state in Q:
# 	print state, Q[state]

plt.plot(range(nbEpisodes), rewardHistory, range(nbEpisodes), explorationHistory, range(nbEpisodes), [195]*nbEpisodes)
plt.ylabel('Number of rewards')
plt.show()
