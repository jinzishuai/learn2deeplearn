#!/usr/bin/python
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt



######################### Define the model ########################
## define the dimensions
nn_input_dim = 1 
nn_output_dim = 1
nn_hdim = 1 #Hidden node dimension
# Create 100 phony x, y data points in NumPy, y = x * 0.1 + 0.3
num_examples = 100
x = tf.Variable(tf.random_uniform([num_examples,nn_input_dim]))
y_ = x**3 + x**2 + x
W1 = tf.Variable(tf.zeros([nn_input_dim, nn_hdim]))
b1 = tf.Variable(tf.zeros([nn_hdim]))
W2 = tf.Variable(tf.zeros( [nn_hdim, nn_output_dim]))
b2 = tf.Variable(tf.zeros([nn_output_dim]))
## define forward propogation
z1 = tf.matmul(x, W1) + b1
#for a list of available tensor flow activation function, see https://www.tensorflow.org/versions/r0.10/api_docs/python/nn.html
#a1 = tf.tanh(z1)
a1 = tf.sigmoid(z1)
y = tf.matmul(a1, W2) + b2
## Define loss and optimizer
# Minimize the mean squared errors.
loss = tf.reduce_mean(tf.square(y - y_))
optimizer = tf.train.GradientDescentOptimizer(0.001)
train = optimizer.minimize(loss)
######################### End of model definition ########################

# Before starting, initialize the variables.  We will 'run' this first.
init = tf.initialize_all_variables() 

# Launch the graph.
sess = tf.Session()
sess.run(init)

plt.plot(x,  y_,  'ro')
plt.show()

# Fit the line.
for step in range(200001):
    sess.run(train)
    if step % 10000 == 0:
        print(step, sess.run(W1), sess.run(b1), sess.run(W2), sess.run(b2),  sess.run(loss))
    if step == 200000:
        plt.plot(x,  y_,  'ro')
        plt.plot(x,  sess.run(y),   'b*')
        plt.show()

