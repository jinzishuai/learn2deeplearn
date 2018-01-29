import gym

import numpy 	

y = .97 #Discount Rate
learnRate = .2
totalEps = 1000

def updateQMat(q, reward, state, action, newState):
	futureReward = max(q[newState][:])
	q[state][action] = q[state][action] + learnRate * (reward + y * futureReward - q[state][action])
	return

success = False
lastFailEp = -1
firstSuccEp = -1
successEps = 0

env = gym.make('FrozenLake-v0')

#env = wrappers.Monitor(env, '/tmp/recording', force=True) #Records performance data

qMatrix = numpy.zeros((env.observation_space.n, env.action_space.n)) #Initialize qMatrix to 0s

for i in range(totalEps):
	observation = env.reset()
	#Loop through episode, one timestep at a time
	for t in range(env.spec.tags.get('wrapper_config.TimeLimit.max_episode_steps')):
		#Create an array of random estimated rewards representing each action
		#with the possible range of rewards decreasing with each episode
		randomActions = numpy.random.randn(1, env.action_space.n)*(1/(i+1))
		#Choose either the action with max expected reward, or a random action
		#according to randomActions array. With each episode, the random actions
		#will become less chosen.
		action = numpy.argmax(qMatrix[observation][:] + randomActions)
		oldObservation = observation;
		observation, reward, done, info = env.step(action) #Perform the action
		if done and reward == 0:
			reward = -1 #Edit reward to negative in the case of falling in a hole
		updateQMat(qMatrix, reward, oldObservation, action, observation) #Update the Q-Matrix
		#env.render()
		if done:
			if reward == 1:
				successEps += 1
				if success == False:
					success = True
					firstSuccEp = i
			else:
				lastFailEp = i
			#print("Episode finished after {} timesteps".format(t+1))
			break

print()
print("Percentage of successful episodes: " + str(successEps/totalEps * 100) + "%")
print()
print("First successful episode: " + str(firstSuccEp))
print()
print("Last failed episode: " + str(lastFailEp))
env.close()
print("qMatrix=%s\n" % qMatrix)
print("Policy is %s\n" % numpy.argmax(qMatrix,axis=1).reshape(4,4))
#gym.upload('/tmp/recording', api_key='sk_fVhBRLT7S7e4MoHswIH5wg') #Uploads performance data