import gym
from gym.wrappers.monitoring import Monitor
from random import randint
import math
import matplotlib.pyplot as plt


def learn(episodeCount):
	for i_episode in range(episodeCount):
		done = False
		obs = env.reset()
		nextState  = getState(obs)
		nextAction = policy(nextState, i_episode)
		while not done:
			# TAKE STEP
			state  = nextState
			action = nextAction
			obs, reward, done, _ = env.step(action)
			nextState  = getState(obs)
			nextAction = policy(nextState, i_episode)
			rewardsHistory[i_episode] = reward

			# UPDATE Q
			discount = 0.9
			alpha = 0.9
			d = reward + discount * Q[nextState][maxAction(Q[nextState])] - Q[state][actionStr]
			Q[state][actionStr] = alpha * d


def getState(obs):
	state = ''
	for o in obs:
		state += str(math.floor(o))
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

stepsHistory       = [0]*nbEpisodes
explorationHistory = [0]*nbEpisodes

env = gym.make('LunarLander-v2')
# env = Monitor(env, 'tmp/cart-pole', force=True)
print env.action_space
print env.observation_space

Q = {}

learn(nbEpisodes)
env.close()

plt.plot(range(nbEpisodes), stepsHistory, range(nbEpisodes), explorationHistory, range(nbEpisodes), [195]*nbEpisodes)
plt.ylabel('Number of rewards')
plt.show()
