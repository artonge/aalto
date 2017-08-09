import gym
import numpy as np
import sys

env = gym.make('FrozenLake-v0')
Q = np.zeros((env.observation_space.n, env.action_space.n))
y = 0.95
num_episodes = 5000
alpha = 0.1
# episode_rewards = np.zeros(num_episodes)

# for i in range(num_episodes):
#     state = env.reset()
#     done = False
#
#     while not done:
#         if np.random.rand() < alpha:
#             action = env.action_space.sample()
#         else:
#             action = np.argmax(Q[state])
#
#         next_state, reward, done, _ = env.step(action)
#         episode_rewards[i] += reward
#         if done:
#             r = 1.0 if reward > 0.0 else -1.0
#         else:
#             r = -0.01
#         Q[state, action] +=  0.1 * (r + y * np.max(Q[next_state]) - Q[state, action])
#         state = next_state
#
#     nb = max(i-100, 0)
#     sys.stdout.write("\rEpisode " + str(nb) + " to " + str(i) + ", medium score: " + str(sum(episode_rewards[nb:i])/(i-nb+1)))
#     sys.stdout.flush()




def learn():
	rList = []
	alpha = np.log(0.000001)/num_episodes
	for i in range(num_episodes):
	    # lr = 0.99**i
		lr = np.exp(alpha*i)
	    #Reset environment and get first new observation
	    state = env.reset()
	    rAll = 0
	    done = False
	    #The Q-Table learning algorithm
	    while not done:
	        #Choose an action by greedily (with noise) picking from Q table
	        if np.random.rand() < lr*0.1:
	            action = np.random.randint(env.action_space.n)
	        else:
	            action = np.argmax(Q[state,:])
	        #Get new state and reward from environment
	        nextstate, reward, done, _ = env.step(action)
	        if done:
	            r = 1.0 if reward > 0.0 else -1.0
	        else:
	            r = -0.01
	        #Update Q-Table with new knowledge
	        Q[state, action] = Q[state, action] +  lr*(r + y * np.max(Q[nextstate,:]) - Q[state, action])
	        rAll += reward
	        state = nextstate

	    rList.append(rAll)

	return sum(rList[-100:])/100.0

sum([learn() for i in range(10)])/10


print "Score over time: " + str(sum([learn() for i in range(10)])/10)
