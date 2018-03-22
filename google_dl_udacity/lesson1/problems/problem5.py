# These are all the modules we'll be using later. Make sure you can import them
# before proceeding further.
from __future__ import print_function
import data

def look_for_duplicates (dataset):
    print ('looking for duplicates in a dataset of shape: ', dataset.shape)


train_dataset, train_labels, valid_dataset, valid_labels, test_dataset, test_labels = data.load_all_data_from_single_pickle_file('../notMNIST.pickle')
look_for_duplicates (valid_dataset)

