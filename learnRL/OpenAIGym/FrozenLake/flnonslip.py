#!/usr/bin/python2.7
import gym
import numpy as np
from gym.envs.registration import register
register(
    id='FrozenLakeNotSlippery-v0',
    entry_point='gym.envs.toy_text:FrozenLakeEnv',
    kwargs={'map_name' : '4x4', 'is_slippery': False},
    max_episode_steps=100,
    reward_threshold=0.78, # optimum = .8196
)
env = gym.make('FrozenLakeNotSlippery-v0')
#Initialize table with all zeros
X = -1
policy=np.array(
        [[1, 2, 2, 0],
         [1, X, 1, X],
         [2, 1, 1, X],
         [X, 2, 2, X]]
        )

policy=policy.flatten()
# Set learning parameters
num_episodes = 1000
#create lists to contain total rewards and steps per episode
#jList = []
results = [0]*num_episodes
for i in range(num_episodes):
    #Reset environment and get first new observation
    s = env.reset()
    #print("initial setup (step 0):")
    #env.render()
    rAll = 0
    d = False
    j = 0
    #The Q-Table learning algorithm
    while j < 99:
        j+=1
        #Get new state and reward from environment
        s1,r,d,_ = env.step(policy[s])
        #print(policy[s])
        #print("%d -> %s" % (s, s1))
        #env.render()
        s = s1
        if d == True:
            if r == 1.0:
                results[i]=1
            else:
                results[i]=0

            print("episod %d finishes in %d steps with result=%d" % (i, j, results[i]) )
            break

#print(results)
print("%d out of %d runs were successful" % (np.sum(results), num_episodes))
