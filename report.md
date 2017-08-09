# Report on openai.gym usability as a student
As part of my internship, Alex Jung asked me to use the [OpenAI Gym](gym.openai.com) platform. The goal was to figure out what knowledge a computer science student would need before being able to use that platform in a reasonable manner. The result will feed his new course on artificial intelligence.


## What I now
**Could have been nice to have**
- Overview of what machine learning is and how it works

Deep learning is a kind of AI base on how humans learn. It consists of layers which will, one after the other, transform the data and produce an output. Each layer passes its output to the next layer. The final output is our answer. Those layers are mimicking neurones and how a human brain works.
- vision: We feed a system lots of data telling him what those are. Later we ask him if a new piece of data is of the same kind than the previous data.
- behavior: Given so observations, we ask a system to make operations to achieve a goal. The system can then learn what suit of operations make it closer or further from the goal. The more it practices, the better it gets.


## First use, with little knowledge on machine learning
### Installation
The installation and use are easy for anyone with basic computer science knowledge. You'll need to know the python programming language and how to install packages.
After installing the following dependencies: `pip`, `cmake`, `swig`, and running: `pip install gym[all]`, you can use it like so:

### Understanding of the code
**Could have been nice to have**
- A working example base on an environment of OpenAI Gym. If possible without an external library

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
        # Compute actions
        # Trivial here, but should be the output of a machine learning system
        action = env.action_space.sample()
        observation, reward, done, info = env.step(action) # Apply actions to the environment
```

#### Step's return value
This part concern how I understand the examples given on the website.
In the Gym's documentations, they first talk about the `env.step` method and what it returns.
- *Observation*: Gives informations about the environment's state and the phenomenon we are watching. This it what we'll use to feed our machine learning system.
- *Reward*: Used to tell our system how good its output is.
- *Done*: If true, the observation is over and the environment needs to be reset.
- *Info*: Contains information about the environment and the reason of its state changes. We're not supposed to use those to train our system !

#### Step's arguments
We can see what kind of arguments (action) we can pass to the `env.step` method. How much action the environment needs and what are their boundaries.

#### Understanding of the framework
OpenAI Gym gives use lots of environments. We ask the environment to execute actions. The environment return us observations and reward. This meaning how the action altered the environment and how much they help to reach the goal.
With OpenAI we're supposed to code the agent. The agent is the machine learning system. That's why we can use any machine learning library we want.
```
  ______>______action______>______
  |                               |
agent                       environment
  |                               |
  ---<---observation/reward---<---
```

#### Others
- I can create environments
- I can record results


## Actual using of OpenAI Gym with Tensorflow
**Could have been nice to have**
- What is a tensor

I could not use the Gym as I had no practical knowledge on how to program machine learning systems, so I followed the tutorial of the Tensorflow library hoping to gain some. This tentative failed as I to little theoretical knowledge and Tensorflow seems to work like magic.


## What I learned
- You give a value to each states. The objective is to be in the highest value state. What gives you the value for a given state is your **value function**.

- You'll build your value function from experience. The environment will sometime give you a reward when you behave correctly. Using one of the following: **MC**, **TD**, **SARSA**, **Q-Learning**, **FA**, you'll update some states value.

- To find the highest value state you've seen, you use your experience. This is called **exploitation**. But to find the best state of all, you need some **exploration**. The balance between exploitation and exploration is defined be the **policy**. The policy dictate you behaviour. You can be greedy, which is exploitation or random which lead to exploration. You can also be e-gready, where you'll be random part of the time.


## Second use, with theoretical knowledge
- Why some algorithms don't work on some environments ?
I tried to implement TD, SARSA, Q-Learning, Eligibility traces on various environments without success.
Only MC seems to work but only when the environment state is discrete or not to complicated.

- How to read the learning curve and improve the algorithm buy tweaking the parameters.
I've heard about over-fitting but not sure how to detect and prevent it.
How do I know my algorithm is exploring to much/no enough.

- Function approximation seems to be impossible without a library to support the core computing. Didn't find an example that show the manipulation of the weights, update and prediction. I tried to implement FA using polynomials, without success. I don't grasp fully how it works.
