#!/usr/bin/python2.7
import gym
env = gym.make('FrozenLake-v0')
env.reset()
for _ in range(10):
    env.render()
    env.step(env.action_space.sample()) # take a random action
