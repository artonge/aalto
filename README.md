# Report on openai.gym usability as a student
As part of my internship, Alex Jung asked me to use the [OpenAI Gym](gym.openai.com) platform. The goal was to figure out what knowledge a computer science student would need before being able to use that platform in a reasonable manner. The result will feed his new course on artificial intelligence.


# Summary
- ### [What I knew](https://github.com/artonge/aalto#what-i-knew-1)
- ### [First use](https://github.com/artonge/aalto#first-use-with-little-knowledge-on-machine-learning)
- ### [Note on Sutton's book](https://github.com/artonge/aalto#notes-on-suttons-book)
- ### [Sudoku environment](https://github.com/artonge/aalto#sudoku-environment-1)


# What I knew
**Naive knowledge**
Deep learning is a kind of AI base on how humans learn. It consists of layers which will, one after the other, transform the data and produce an output. Each layer passes its output to the next layer. The final output is our answer. Those layers are mimicking neurones and how a human brain works.
- **vision**: We feed a system lots of data telling him what those are. Later we ask him if a new piece of data is of the same kind than the previous data.
- **behavior**: Given some observations, we ask a system to make operations to achieve a goal. The system can then learn what suit of operations make it closer or further from the goal. The more it practices, the better it gets.

**Could have been nice to have**
- Overview of what machine learning is and how it works


# First use, with little knowledge on machine learning
## Installation
The installation and use are easy for anyone with knowledge of python and pip. You can use it with python2 or 3.
After installing the following dependencies: `pip`, `cmake`, `swig`, and running: `pip install gym[all]`, you can use it like so:

```python
# Import gym library
import gym

# Create a new CartPole-v0 environment
env = gym.make('CartPole-v0')
# Learn for twenty episodes
for i_episode in range(20):
	# Reset the environment
	# Resetting gives us the first observation
	observation = env.reset()
	# Take step until the episode is done
	# The episode is done when the environment tells us it's done
	done = False
	while not done:
		# Display the state of the environment
		env.render()
		# Random here, but should be the output of a machine learning algorithm
		action = env.action_space.sample()
		# Take the action
		observation, reward, done, info = env.step(action)
```

**Could have been nice to have**
- A working example based on an environment of OpenAI Gym. If possible without an external library

## The framework
```
   ___>_____(action)_____>______
  |                             |
[agent]                      [environment]
  |                             |
  |__<__(observation/reward)_<__|
```
The gym allow us the only code the agent, and to forget about the environment. Therefore we can easily use external libraries like Tensorflow.
It gives us information about the type of the observations and actions for a given environment.

## Step's return value
In the Gym's documentations, they first talk about the `env.step` method and what it returns.
- **Observation**: Gives informations about the environment's state and the phenomenon we are watching. This it what we'll use to feed our machine learning system.
- **Reward**: Used to tell our system how good the environment state is.
- **Done**: If true, the observation is over and the environment needs to be reset.
- **Info**: Contains information about the environment and the reason of its state changes. We're not supposed to use this to train our system !

## Others
- You can create environments, I did one for the [sudoku](https://github.com/artonge/gym-sudoku)
- You can record results, upload them on their servers and [share](https://gym.openai.com/evaluations/eval_pSQ46cQKRKOEMQSFPp3EA#reproducibility) them with any one.

## Attempt to use the gym with Tensorflow
I could not use the Gym as I had no knowledge on how to program machine learning systems, so I followed the tutorial of the Tensorflow library hoping to gain some. This tentative failed and Tensorflow seems to work with magic.

*I was thinking that it was mandatory to use a library like Tensorflow to have any results in reinforcement learning, but you can go far without.*


# Notes on [Sutton's Book](http://incompleteideas.net/sutton/book/bookdraft2017june.pdf)
I didn't went further than function approximation. This is because I wanted to fully understand function approximation before using deep reinforcement learning. But unfortunately I didn't understand it as much as I wanted. I only found one [example](https://github.com/dennybritz/reinforcement-learning/blob/master/FA/Q-Learning%20with%20Value%20Function%20Approximation%20Solution.ipynb) and it was using an external library to do all the work, so it wasn't helpful.

## Overview
- You code the **agent** that will interact with the **environment**. The environment gives you the state he is in, and you need to find the best action for that state.

- To find the best action, you use the **value function**. It map a state to a value.

- The value function is built from experience. The environment will sometimes gives you a **reward** when you behave correctly. Then using one of the following: **TD(λ)**, **MC**, **SARSA**, **Q-Learning**, **FA**, you'll update some states value.

- To find the highest value state you've seen, you use your experience. This is called **exploitation**. But to find the best state of all, you need some **exploration**. The balance between exploitation and exploration is defined be the **policy**. The policy dictate your behavior. You can be **greedy**, which is exploitation or random which lead to exploration. You can also be **e-greedy**, where you'll be random part of the time.

- The learning process is composed of **episodes**, and each episode is composed of **steps**. The environment is reset between all episodes. Each steps is composed of a state, an action to go to the next state, and a reward received when arriving in the state.

- The **return** of a state is the cumulated reward received from the first encounter to the end of the episode.

## Methods to update the value function
- **Temporal Difference** - TD(λ) : Use the return to update the value of a state. In TD(λ) the return is discounted, using the discount factor λ. This factor will progressively decrease the impact of a reward on the previous states. The farther the state from the reward, the smaller the impact on its value.

- **TD(0)** : Is a special case of TD, where λ is 0. So the value update only take into account the next state.

- **Monte Carlo** - MC : Is a special case of TD, where λ is 1. So the return is not discounted.

- **State Action Reward State Action** - SARSA : Instead of considering the value of a state, you consider the value of a state-action pair.

- **Q-Learning** : Instead of using a policy based on your value function, you use a policy base on something else. This is called **Off-Policy** control and the classic methods are called **On-Policy**. For example, during learning you use a random policy to maximize the exploration and when you want to exploit what you learned, you use an e-policy based on the learned function value.

- **Eligibility Traces** : Allows TD to update the value during the episode. This is called **On-Line** and when you update the value at the end of the episode it is called **Off-line**. Each states has a trace, this trace augment when the state is visited. At each steps, all the traces are decayed. Then at each steps, you update the value of each state using the reward for the current state, and the trace.

- **Function Approximation** - FA : Instead of a discrete value function, build a continuous value function using curve fitting.

## Implementation of algorithms
- The [CartPole](https://github.com/artonge/aalto/tree/master/CartPole): folder contains basics examples only on the CartPole environment.
- The [Exercises](https://github.com/artonge/aalto/tree/master/Exercises): folder contains some solutions to the exercises found in this [repository](https://github.com/dennybritz/reinforcement-learning).
- The [Sandbox](https://github.com/artonge/aalto/tree/master/Sandbox) folder contains attempts to solve some environments.

### Problems I encountered
- Why some algorithms don't work on some environments ?
I tried to implement TD, SARSA, Q-Learning, Eligibility traces on various environments without success.
Only MC seems to work but only when the environment state is discrete or not to complicated.

- How to read the learning curve and improve the algorithm buy tweaking the parameters.
I've heard about over-fitting but not sure how to detect and prevent it.
How do I know my algorithm is exploring to much/no enough.

- Function approximation seems to be impossible without a library to support the core computing. Didn't find an example that show the manipulation of the weights, update and prediction. I tried to implement FA using polynomials, without success.

### Notation
- V : Value function
- Q : state-action value function
- G : Return


# [Sudoku environment](https://github.com/artonge/gym-sudoku)
I heard the sudoku was considered for teaching, so I made an sudoku environment for OpenAI gym. After testing it I think it's not the best environment.

1. the grid generation can take some times so it slows the attempts a student can make. This could probably be improved.
2. due to the amount of possible states, it's a hard environment, so not perfect for a first contact with reinforcement learning. Maybe reducing the size of the grid could make it easier.

The [Sudoku](https://github.com/artonge/aalto/tree/master/Sudoku) folder contains two trials, but unsuccessful.

I would recommended the CartPole environment for teaching, because the amount of states is tinier than for the sudoku. Or even the [tic-tac-toe](https://en.wikipedia.org/wiki/Tic-tac-toe) game, but this one is maybe too easy.

I can give you access to the gym-sudoku repository on github so you can make modifications, and to the pip account so you can publish them.
