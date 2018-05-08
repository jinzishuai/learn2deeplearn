import loadData
import numpy as np
import tensorflow as tf

batch_size = 16
patch_size = 5
depth = 16
num_hidden = 64

train_dataset, train_labels,valid_dataset, valid_labels,test_dataset, test_labels,image_size, num_labels, num_channels = loadData.load_all_data_from_single_pickle_file('../notMNIST.pickle')
graph = tf.Graph()

with graph.as_default():

  # Input data.
  tf_train_dataset = tf.placeholder(
    tf.float32, shape=(batch_size, image_size, image_size, num_channels))
  tf_train_labels = tf.placeholder(tf.float32, shape=(batch_size, num_labels))
  tf_valid_dataset = tf.constant(valid_dataset)
  tf_test_dataset = tf.constant(test_dataset)
  
  # Variables.
  layer3_weights = tf.Variable(tf.truncated_normal(
      [image_size // 2 * image_size // 2 * num_channels, num_hidden], stddev=0.1))
  layer3_biases = tf.Variable(tf.constant(1.0, shape=[num_hidden]))
  layer4_weights = tf.Variable(tf.truncated_normal(
      [num_hidden, num_labels], stddev=0.1))
  layer4_biases = tf.Variable(tf.constant(1.0, shape=[num_labels]))
  
  # Model.
  def model(data):
    # Data is shaped of [batch_size, image_size, image_size, num_channels]
    hidden = tf.nn.max_pool(data, [1, 2, 2, 1],[1, 2, 2, 1] , padding='SAME') #same shape of [batch_size, image_size/2, image_size/2, num_channels]
    shape = hidden.get_shape().as_list() 
    reshape = tf.reshape(hidden, [shape[0], shape[1] * shape[2] * shape[3]]) #reshaped into 2D array of [batch_size, image_size/2* image_size/2 * num_channels]
    hidden = tf.nn.relu(tf.matmul(reshape, layer3_weights) + layer3_biases)
    return tf.matmul(hidden, layer4_weights) + layer4_biases
  
  # Training computation.
  logits = model(tf_train_dataset)
  loss = tf.reduce_mean(
    tf.nn.softmax_cross_entropy_with_logits(labels=tf_train_labels, logits=logits))
    
  # Optimizer.
  optimizer = tf.train.GradientDescentOptimizer(0.05).minimize(loss)
  
  # Predictions for the training, validation, and test data.
  train_prediction = tf.nn.softmax(logits)
  valid_prediction = tf.nn.softmax(model(tf_valid_dataset))
  test_prediction = tf.nn.softmax(model(tf_test_dataset))


# In[6]:


num_steps = 1001

with tf.Session(graph=graph) as session:
  tf.global_variables_initializer().run()
  print('Initialized')
  for step in range(num_steps):
    offset = (step * batch_size) % (train_labels.shape[0] - batch_size)
    batch_data = train_dataset[offset:(offset + batch_size), :, :, :]
    batch_labels = train_labels[offset:(offset + batch_size), :]
    feed_dict = {tf_train_dataset : batch_data, tf_train_labels : batch_labels}
    _, l, predictions = session.run(
      [optimizer, loss, train_prediction], feed_dict=feed_dict)
    if (step % 50 == 0):
      print('Minibatch loss at step %d: %f' % (step, l))
      print('Minibatch accuracy: %.1f%%' % loadData.accuracy(predictions, batch_labels))
      print('Validation accuracy: %.1f%%' % loadData.accuracy(
        valid_prediction.eval(), valid_labels))
  print('Test accuracy: %.1f%%' % loadData.accuracy(test_prediction.eval(), test_labels))

