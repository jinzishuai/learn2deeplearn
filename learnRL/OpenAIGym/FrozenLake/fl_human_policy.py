#!/usr/bin/python2.7
import gym
import numpy as np
env = gym.make('FrozenLake-v0')
#Initialize table with all zeros
X = -1
X = -1
policy=np.array(
        [[0, 3, 3, 3],
         [0, X, 2, X],
         [3, 1, 0, X],
         [X, 2, 1, X]]
        )
print("policy=\n%s\n" % policy)
policy=policy.flatten()
# Set learning parameters
num_episodes = 10000
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
        #env.render()
        s1,r,d,_ = env.step(policy[s])
        #print("%d -> %s" % (s, s1))
        #env.render()
        s = s1
        if d == True:
            if r == 1.0:
                results[i]=1
            else:
                results[i]=0

            break

#print(results)
print("%d out of %d runs were successful" % (np.sum(results), num_episodes))
