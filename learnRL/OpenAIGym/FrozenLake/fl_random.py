#!/usr/bin/python2.7
import gym
env = gym.make('FrozenLake-v0')
env.reset()
print("%s observations" % env.observation_space)
print("%s actions" % env.action_space)

env.render()
for _ in range(3):
    env.step(env.action_space.sample()) # take a random action
    env.render()
