#!/usr/bin/python
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# Create 100 phony x, y data points in NumPy, y = x * 0.1 + 0.3
x_data = np.random.rand(100).astype(np.float32)
y_data = x_data**3 + x_data**2 + x_data
plt.plot(x_data,  y_data,  'ro')

W1 = tf.Variable(tf.random_uniform([1], -1.0, 1.0))
b1 = tf.Variable(tf.zeros([1]))
W2 = tf.Variable(tf.random_uniform([1], -1.0, 1.0))
b2 = tf.Variable(tf.zeros([1]))

z1 = W1 * x_data + b1
#for a list of available tensor flow activation function, see https://www.tensorflow.org/versions/r0.10/api_docs/python/nn.html
#a1 = tf.tanh(z1)
a1 = tf.sigmoid(z1)
z1 = a1 * W2 + b2
y = z1

# Minimize the mean squared errors.
loss = tf.reduce_mean(tf.square(y - y_data))
optimizer = tf.train.GradientDescentOptimizer(0.001)
train = optimizer.minimize(loss)

# Before starting, initialize the variables.  We will 'run' this first.
init = tf.initialize_all_variables()

# Launch the graph.
sess = tf.Session()
sess.run(init)

# Fit the line.
for step in range(200001):
    sess.run(train)
    if step % 10000 == 0:
        print(step, sess.run(W1), sess.run(b1), sess.run(W2), sess.run(b2),  sess.run(loss))
    if step == 200000:
        plt.plot(x_data,  sess.run(y),   'b*')
        plt.show()

