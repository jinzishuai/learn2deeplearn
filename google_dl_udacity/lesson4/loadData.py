
from __future__ import print_function
from six.moves import cPickle as pickle
from six.moves import range



import numpy as np
 

def load_all_data_from_single_pickle_file(pickle_file='../notMNIST.pickle'):
    image_size = 28
    num_labels = 10
    num_channels = 1 # grayscale
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

    def reformat(dataset, labels):
      dataset = dataset.reshape(
        (-1, image_size, image_size, num_channels)).astype(np.float32)
      labels = (np.arange(num_labels) == labels[:,None]).astype(np.float32)
      return dataset, labels
    train_dataset, train_labels = reformat(train_dataset, train_labels)
    valid_dataset, valid_labels = reformat(valid_dataset, valid_labels)
    test_dataset, test_labels = reformat(test_dataset, test_labels)
    print('Training set', train_dataset.shape, train_labels.shape)
    print('Validation set', valid_dataset.shape, valid_labels.shape)
    print('Test set', test_dataset.shape, test_labels.shape)

    return train_dataset, train_labels,valid_dataset, valid_labels,test_dataset, test_labels, image_size, num_labels, num_channels


def accuracy(predictions, labels):
  return (100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1))
          / predictions.shape[0])


