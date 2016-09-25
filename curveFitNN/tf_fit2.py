#!/usr/bin/python
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


sess = tf.InteractiveSession()

######################### Define the model ########################
## define the dimensions
nn_input_dim = 1 
nn_output_dim = 1
nn_hdim = 1 #Hidden node dimension
# Create 100 phony x, y data points in NumPy, y = x * 0.1 + 0.3
num_examples = 100

x = tf.placeholder(tf.float32, [None, nn_input_dim])
y_ = tf.placeholder(tf.float32, [None, nn_output_dim])

x_data = np.random.rand(num_examples, nn_input_dim).astype(np.float32)
y_data = x_data**3 + x_data**2 + x_data
plt.plot(x_data,  y_data,  'ro')


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
train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss)
######################### End of model definition ########################


tf.initialize_all_variables().run()
NSteps=10000
NReport = NSteps/10
for step in range(NSteps+1):
    train_step.run({x:x_data,  y_:y_data})
    if step % NReport == 0:
        print(step, loss.eval({x:x_data,  y_:y_data}))

# # Test trained model
#correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
#accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
#print(accuracy.eval({x: mnist.test.images, y_: mnist.test.labels})
x_test = np.linspace(0.0,  1.0,  1000).reshape(1000, 1)
plt.plot(x_test,  y.eval({x:x_test}) ,  'b-')
plt.show()
