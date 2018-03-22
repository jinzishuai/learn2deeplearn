# These are all the modules we'll be using later. Make sure you can import them
# before proceeding further.
from __future__ import print_function
import pickle

def load_all_data_from_single_pickle_file(file):
    with open(file, 'rb') as fp:
        all_data = pickle.load(fp)

    train_dataset = all_data['train_dataset']
    train_labels = all_data['train_labels']
    valid_dataset = all_data['valid_dataset']
    valid_labels = all_data['valid_labels']
    test_dataset = all_data['test_dataset']
    test_labels = all_data['test_labels']
    print('Training shapes:', train_dataset.shape, train_labels.shape)
    print('Validation shapes:', valid_dataset.shape, valid_labels.shape)
    print('Testing shapes:', test_dataset.shape, test_labels.shape)
    return train_dataset, train_labels, valid_dataset, valid_labels, test_dataset, test_labels
