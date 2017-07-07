import gym
from gym.wrappers.monitoring import Monitor
from random import randint
import time
import math
import matplotlib.pyplot as plt


def learn(episodeCount):
	for i_episode in range(episodeCount):
		# Start an episode
		observation = env.reset()
		tmpHis = {}
		totalRewards = 0
		
		for t in range(200):
			state, sState = getState(observation) # Get the state
			action = policy(state, i_episode) # Get the action
			tmpHis[sState] = action # Add state and action to temHistory
			observation, reward, done, info = env.step(action) # Apply the action
			totalRewards += reward # Update total reward for this episode
			if done: #Episode is over
				for sState, action in tmpHis.iteritems(): # Update value for choosen actions
					a = history[sState][action]					
					a['count'] = a['count'] + 1
					a['value'] = (a['value'] * a['count'] + totalRewards) / (a['count'] + 1)
				if i_episode % 100 == 0:
					print i_episode, t+1
				break


def getState(obs):
	sState = str(math.floor(obs[0]))+str(math.floor(obs[1]))+str(math.floor(obs[2]))+str(math.floor(obs[3]))
	if not history.has_key(sState):
		history[sState] = [{'count':0, 'value':0}, {'count':0, 'value':0}]
	return history[sState], sState


def policy(state, episode):
	if randint(0, 100) < max(-math.exp(float(episode)/300)+70, 0):
		return env.action_space.sample()
	else:
		if state[0]['value']> state[1]['value']:
			return 0
		else:
			return 1


history = {}
env = gym.make('CartPole-v0')
#env = Monitor(env, 'tmp/cart-pole', force=True)
learn(3000)
env.close()
#gym.upload('tmp/cart-pole', api_key='sk_QoYvL963TwnAqSJXZLOQ')

