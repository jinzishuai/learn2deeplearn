import gym
import numpy as np
import random
import tensorflow as tf
import matplotlib.pyplot as plt


env = gym.make('FrozenLake-v0')

tf.reset_default_graph()

#These lines establish the feed-forward part of the network used to choose actions
inputs1 = tf.placeholder(shape=[1,16],dtype=tf.float32)
W = tf.Variable(tf.random_uniform([16,4],0,0.1))
Qout = tf.matmul(inputs1,W)
predict = tf.argmax(Qout,1)

#Below we obtain the loss by taking the sum of squares difference between the target and prediction Q values.
nextQ = tf.placeholder(shape=[1,4],dtype=tf.float32)
loss = tf.reduce_sum(tf.square(nextQ - Qout))
trainer = tf.train.GradientDescentOptimizer(learning_rate=0.1)
updateModel = trainer.minimize(loss)

init = tf.global_variables_initializer()

#saver = tf.train.Saver()
#export_dir = "export"
#builder = tf.saved_model.builder.SavedModelBuilder(export_dir)
# Set learning parameters
y = .99
e = 0.5
initE = e
num_episodes = 2000
#create lists to contain total rewards and steps per episode
lossList = []
rList = []
with tf.Session() as sess:
    sess.run(init)
    for i in range(num_episodes):
        #Reset environment and get first new observation
        s = env.reset()
        rAll = 0
        d = False
        j = 0
        #The Q-Network
        while j < 99:
            j+=1
            #Choose an action by greedily (with e chance of random action) from the Q-network
            a,allQ = sess.run([predict,Qout],feed_dict={inputs1:np.identity(16)[s:s+1]})
            if np.random.rand(1) < e:
                a[0] = env.action_space.sample()
            #Get new state and reward from environment
            s1,r,d,_ = env.step(a[0])
            #Obtain the Q' values by feeding the new state through our network
            Q1 = sess.run(Qout,feed_dict={inputs1:np.identity(16)[s1:s1+1]})
            #Obtain maxQ' and set our target value for chosen action.
            maxQ1 = np.max(Q1)
            targetQ = allQ
            targetQ[0,a[0]] = r + y*maxQ1
            #Train our network using target and predicted Q values
            _,W1 = sess.run([updateModel,W],feed_dict={inputs1:np.identity(16)[s:s+1],nextQ:targetQ})
            if d == True: lossList.append(sess.run(loss, {nextQ:targetQ, inputs1:np.identity(16)[s:s+1]}))
            rAll += r
            s = s1
            #env.render()
            if d == True:
                #Reduce chance of random action as we train the model.
                #print("Episode finished after " + str(j) + " timesteps")
                #e = 1./((i/50) + 10) #Not agressive enough, sometimes never finds goal
                e = initE - (i/num_episodes)
                break
        rList.append(rAll)
    print("saving model...")
    #saver.save(sess, './flmodel')
    print("W=%s\n" % sess.run([W]))
    print("policy=%s\n" % sess.run([tf.reshape(tf.argmax(W,1), [4, 4])]))
    #builder.add_meta_graph_and_variables(sess, ["TRAINED"])									   
print("Percent of successful episodes: " + str((sum(rList)/num_episodes)*100) + "%")

plt.plot(rList)
#builder.save()
#plt.plot(lossList)


plt.show()