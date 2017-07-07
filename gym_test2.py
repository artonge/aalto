import gym
from gym.wrappers.monitoring import Monitor
from random import randint
import time
import math
import matplotlib.pyplot as plt
import numpy as np


def learn(episodeCount, i_try):
	for i_episode in range(episodeCount): # Start an episode
		observation = env.reset()
		tmpHis = []
		totalRewards = 0
		
		for t in range(1000):
			state, sState = getState(observation) # Get the state
			action = policy(state, i_episode) # Get the action
			tmpHis.append({'state': sState, 'action':action}) # Add state and action to temHistory
			observation, reward, done, info = env.step(action) # Apply the action
			totalRewards += reward # Update total reward for this episode
			if done: #Episode is over
				steps[i_episode] = ((steps[i_episode]*i_try + float(t+1))/(i_try+1))
				for item in tmpHis: # Update value for choosen actions
					a = history[item['state']][item['action']]	
					a['value'] = (a['value'] * a['count'] + totalRewards) / (a['count'] + 1)				
					a['count'] += 1
				if i_episode % 1000 == 0:
					print i_episode, t+1, max(-math.exp(float(i_episode)/50)+30, 0)
				break


def getState(obs):
	#sState = str(obs[0]>0)+str(obs[1]>0)+str(obs[2]>0)+str(obs[3]>0)
	sState = str(math.floor(obs[0]))+str(math.floor(obs[1]))+str(math.floor(obs[2]))+str(math.floor(obs[3]))
	if not history.has_key(sState):
		history[sState] = [{'count':0, 'value':0}, {'count':0, 'value':0}]
	return history[sState], sState


def policy(state, episode):
	#d = max(-math.exp(float(episode)/200)+30, 0)
	d = max(-episode*decay1+decay2, 0)
	decay[episode] = d
	if randint(0, 100) < d:
		return env.action_space.sample()
	else:
		if state[0]['value']> state[1]['value']:
			return 0
		else:
			return 1


episodeCount = 500

OBJECTIF = np.array([195]*episodeCount)
decay1 = 1
decay2 = 20

env = gym.make('CartPole-v0')
#env = Monitor(env, 'tmp/cart-pole', force=True)
oldDelta = 200*episodeCount
steps = [0]*episodeCount
decay = [0]*episodeCount
while True:
	oldSteps = steps
	oldDecay = decay
	steps = [0]*episodeCount
	decay = [0]*episodeCount
	for i in range(50):
		history = {}
		learn(episodeCount, i)
	delta = sum(OBJECTIF - np.array(steps))
	print decay1
	if delta < oldDelta:
		oldDelta = delta
		decay1 -= 0.2
	else:
		decay1 += 0.2
		steps = oldSteps
		decay = oldDecay
		break
while True:
	oldSteps = steps
	oldDecay = decay
	steps = [0]*episodeCount
	decay = [0]*episodeCount
	for i in range(20):
		history = {}
		learn(episodeCount, i)
	delta = sum(OBJECTIF - np.array(steps))
	print decay2
	if delta < oldDelta:
		oldDelta = delta
		decay2 -= 1
	else:
		decay2 += 1
		steps = oldSteps
		decay = oldDecay
		break
print decay1, decay2
env.close()
#gym.upload('tmp/cart-pole', api_key='sk_QoYvL963TwnAqSJXZLOQ')
plt.plot(range(episodeCount), steps, range(episodeCount), OBJECTIF, range(episodeCount), decay)
plt.show()
