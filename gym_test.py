import gym
import tensorflow as tf


env = gym.make('CartPole-v0')
for i_episode in range(1):
    observation = env.reset()
    sess = tf.Session()
    init = tf.global_variables_initializer()
    a = tf.Variable([observation[0]], dtype=tf.float32)
    b = tf.Variable([observation[1]], dtype=tf.float32)
    c = tf.Variable([observation[2]], dtype=tf.float32)
    d = tf.Variable([observation[3]], dtype=tf.float32)
    sess.run(init)
    for t in range(100):
        env.render()
        action = env.action_space.sample()
        observation, reward, done, info = env.step(action)
        if done:
            print("Episode finished after {} timesteps".format(t+1))
            break
