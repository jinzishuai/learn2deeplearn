
# coding: utf-8

# Deep Learning
# =============
# 
# Assignment 2
# ------------
# 
# Previously in `1_notmnist.ipynb`, we created a pickle with formatted datasets for training, development and testing on the [notMNIST dataset](http://yaroslavvb.blogspot.com/2011/09/notmnist-dataset.html).
# 
# The goal of this assignment is to progressively train deeper and more accurate models using TensorFlow.

# In[1]:


# These are all the modules we'll be using later. Make sure you can import them
# before proceeding further.
from __future__ import print_function
import numpy as np
import tensorflow as tf
from six.moves import cPickle as pickle
from six.moves import range


# First reload the data we generated in `1_notmnist.ipynb`.

# In[2]:


pickle_file = '../notMNIST.pickle'

with open(pickle_file, 'rb') as f:
  save = pickle.load(f)
  train_dataset = save['train_dataset']
  train_labels = save['train_labels']
  valid_dataset = save['valid_dataset']
  valid_labels = save['valid_labels']
  test_dataset = save['test_dataset']
  test_labels = save['test_labels']
  del save  # hint to help gc free up memory
  print('Training set', train_dataset.shape, train_labels.shape)
  print('Validation set', valid_dataset.shape, valid_labels.shape)
  print('Test set', test_dataset.shape, test_labels.shape)


# Reformat into a shape that's more adapted to the models we're going to train:
# - data as a flat matrix,
# - labels as float 1-hot encodings.

# In[3]:


image_size = 28
num_labels = 10

def reformat(dataset, labels):
  dataset = dataset.reshape((-1, image_size * image_size)).astype(np.float32)
  # Map 0 to [1.0, 0.0, 0.0 ...], 1 to [0.0, 1.0, 0.0 ...]
  labels = (np.arange(num_labels) == labels[:,None]).astype(np.float32)
  return dataset, labels

def accuracy(predictions, labels):
  return (100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1))
          / predictions.shape[0])


train_dataset, train_labels = reformat(train_dataset, train_labels)
valid_dataset, valid_labels = reformat(valid_dataset, valid_labels)
test_dataset, test_labels = reformat(test_dataset, test_labels)
print('Training set', train_dataset.shape, train_labels.shape)
print('Validation set', valid_dataset.shape, valid_labels.shape)
print('Test set', test_dataset.shape, test_labels.shape)

################## Code from endri.deliu #########################
# ref: https://discussions.udacity.com/t/assignment-4-problem-2/46525/25

batch_size = 128
hidden_layer1_size = 1024
hidden_layer2_size = 305
hidden_lastlayer_size = 75

use_multilayers = True

regularization_meta=0.03


graph = tf.Graph()
with graph.as_default():

  # Input data. For the training data, we use a placeholder that will be fed
  # at run time with a training minibatch.
  tf_train_dataset = tf.placeholder(tf.float32,
                                    shape=(batch_size, image_size * image_size))
  tf_train_labels = tf.placeholder(tf.float32, shape=(batch_size, num_labels))
  tf_valid_dataset = tf.constant(valid_dataset)
  tf_test_dataset = tf.constant(test_dataset)
  
  # Variables.
  keep_prob = tf.placeholder(tf.float32)

  weights_layer1 = tf.Variable(
    tf.truncated_normal([image_size * image_size, hidden_layer1_size], stddev=0.0517))
  biases_layer1 = tf.Variable(tf.zeros([hidden_layer1_size]))

  if use_multilayers:
    weights_layer2 = tf.Variable(
      tf.truncated_normal([hidden_layer1_size, hidden_layer1_size], stddev=0.0441))
    biases_layer2 = tf.Variable(tf.zeros([hidden_layer1_size]))

    weights_layer3 = tf.Variable(
      tf.truncated_normal([hidden_layer1_size, hidden_layer2_size], stddev=0.0441))
    biases_layer3 = tf.Variable(tf.zeros([hidden_layer2_size]))
    
    weights_layer4 = tf.Variable(
      tf.truncated_normal([hidden_layer2_size, hidden_lastlayer_size], stddev=0.0809))
    biases_layer4 = tf.Variable(tf.zeros([hidden_lastlayer_size]))


  weights = tf.Variable(
    tf.truncated_normal([hidden_lastlayer_size if use_multilayers else hidden_layer1_size, num_labels], stddev=0.1632))
  biases = tf.Variable(tf.zeros([num_labels]))
  
    
  # get the NN models
  def getNN4Layer(dSet, use_dropout):
    input_to_layer1 = tf.matmul(dSet, weights_layer1) + biases_layer1
    hidden_layer1_output = tf.nn.relu(input_to_layer1)
    
    
    logits_hidden1 = None
    if use_dropout:
       dropout_hidden1 = tf.nn.dropout(hidden_layer1_output, keep_prob)
       logits_hidden1 = tf.matmul(dropout_hidden1, weights_layer2) + biases_layer2
    else:
      logits_hidden1 = tf.matmul(hidden_layer1_output, weights_layer2) + biases_layer2
    
    hidden_layer2_output = tf.nn.relu(logits_hidden1)
    
    logits_hidden2 = None
    if use_dropout:
       dropout_hidden2 = tf.nn.dropout(hidden_layer2_output, keep_prob)
       logits_hidden2 = tf.matmul(dropout_hidden2, weights_layer3) + biases_layer3
    else:
      logits_hidden2 = tf.matmul(hidden_layer2_output, weights_layer3) + biases_layer3
    
    
    hidden_layer3_output = tf.nn.relu(logits_hidden2)
    logits_hidden3 = None
    if use_dropout:
       dropout_hidden3 = tf.nn.dropout(hidden_layer3_output, keep_prob)
       logits_hidden3 = tf.matmul(dropout_hidden3, weights_layer4) + biases_layer4
    else:
      logits_hidden3 = tf.matmul(hidden_layer3_output, weights_layer4) + biases_layer4
    
    
    hidden_layer4_output = tf.nn.relu(logits_hidden3)
    logits = None
    if use_dropout:
       dropout_hidden4 = tf.nn.dropout(hidden_layer4_output, keep_prob)
       logits = tf.matmul(dropout_hidden4, weights) + biases
    else:
      logits = tf.matmul(hidden_layer4_output, weights) + biases
    
    return logits

  # get the NN models
  def getNN1Layer(dSet, use_dropout, w1, b1, w, b):
    input_to_layer1 = tf.matmul(dSet, w1) + b1
    hidden_layer1_output = tf.nn.relu(input_to_layer1)
        
    logits = None
    if use_dropout:
       dropout_hidden1 = tf.nn.dropout(hidden_layer1_output, keep_prob)
       logits = tf.matmul(dropout_hidden1, w) + b
    else:
      logits = tf.matmul(hidden_layer1_output, w) + b
    
    return logits

  
  
  # Training computation.
  logits = getNN4Layer(tf_train_dataset, True)  
  logits_valid = getNN4Layer(tf_valid_dataset, False)
  logits_test = getNN4Layer(tf_test_dataset, False)
    
  
  loss = tf.reduce_mean(
    tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=tf_train_labels))
  #loss_l2 = loss + (regularization_meta * (tf.nn.l2_loss(weights)))
  
  global_step = tf.Variable(0)  # count the number of steps taken.
  learning_rate = tf.train.exponential_decay(0.3, global_step, 3500, 0.86, staircase=True)
  
    
  # Optimizer.
  optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=global_step)
  
  # Predictions for the training, validation, and test data.
  train_prediction = tf.nn.softmax(logits)
  valid_prediction = tf.nn.softmax(logits_valid)
  test_prediction = tf.nn.softmax(logits_test)



num_steps = 95001

with tf.Session(graph=graph) as session:
  tf.initialize_all_variables().run()
  print("Initialized")
  for step in range(num_steps):
    # Pick an offset within the training data, which has been randomized.
    # Note: we could use better randomization across epochs.
    offset = (step * batch_size) % (train_labels.shape[0] - batch_size)
    
    # Generate a minibatch.
    batch_data = train_dataset[offset:(offset + batch_size), :]
    batch_labels = train_labels[offset:(offset + batch_size), :]
    
    # Prepare a dictionary telling the session where to feed the minibatch.
    # The key of the dictionary is the placeholder node of the graph to be fed,
    # and the value is the numpy array to feed to it.
    feed_dict = {tf_train_dataset : batch_data, tf_train_labels : batch_labels, keep_prob:0.75}
    _, l, predictions = session.run(
      [optimizer, loss, train_prediction], feed_dict=feed_dict)
    if (step % 500 == 0):
      print("Minibatch loss at step", step, ":", l)
      print("Minibatch accuracy: %.1f%%" % accuracy(train_prediction.eval(feed_dict={tf_train_dataset : batch_data, tf_train_labels : batch_labels, keep_prob:1.0}), batch_labels))
      print("Validation accuracy: %.1f%%" % accuracy(
        valid_prediction.eval(feed_dict={keep_prob:1.0}), valid_labels))
  print("##########################")
  print("Test accuracy: %.1f%%" % accuracy(test_prediction.eval(feed_dict={keep_prob:1.0}), test_labels))