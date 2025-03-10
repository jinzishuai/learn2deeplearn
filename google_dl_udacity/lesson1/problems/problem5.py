# These are all the modules we'll be using later. Make sure you can import them
# before proceeding further.
from __future__ import print_function
import matplotlib.pyplot as plt
import numpy as np
import data

def look_for_duplicates (dataset):
    print ('looking for duplicates in a dataset of shape: ', dataset.shape)
    sorted, indices = np.unique(dataset, axis=0, return_index = True)
    print('after unique sort, the shape of dataset is ', sorted.shape)
    print("there are %d duplicate entries out of %d, unique rate of %.2f%%" % ( (dataset.shape[0]-sorted.shape[0]), dataset.shape[0], sorted.shape[0]*100.0/dataset.shape[0]) )
    return sorted, indices

def sort4images(mytest):
    plt.figure(1)
    plt.subplot(221)
    plt.imshow(mytest[0,:,:])
    plt.subplot(222)
    plt.imshow(mytest[1,:,:])
    plt.subplot(223)
    plt.imshow(mytest[2,:,:])
    plt.subplot(224)
    plt.imshow(mytest[3,:,:])

    #mytest.sort(axis=2)
    sorted, indices = np.unique(mytest, axis=0, return_index = True)
    plt.figure(2)
    plt.subplot(221)
    plt.imshow(sorted[0,:,:])
    plt.subplot(222)
    plt.imshow(sorted[1,:,:])
    plt.subplot(223)
    plt.imshow(sorted[2,:,:])
    plt.subplot(224)
    plt.imshow(sorted[3,:,:])

    return sorted, indices

letters=np.array(['A', 'B', 'C','D','E','F','G','H','I','J'])
train_dataset, train_labels, valid_dataset, valid_labels, test_dataset, test_labels = data.load_all_data_from_single_pickle_file('../../notMNIST.pickle')
# sort the train_set
#print('my test results are',letters[train_labels[0:4]])
#sorted, indices = sort4images(train_dataset[0:4,:,:])
#print('my sorted test results are',letters[train_labels[indices]])
#plt.show(block=False)

sorted, indices = look_for_duplicates (valid_dataset)
uniqie_valid_dataset = valid_dataset[indices]
unique_valid_labels = valid_labels[indices]

sorted, indices = look_for_duplicates (test_dataset)
uniqie_test_dataset = test_dataset[indices]
unique_test_labels = test_labels[indices]

sorted, indices = look_for_duplicates (train_dataset)
uniqie_train_dataset = train_dataset[indices]
unique_train_labels = train_labels[indices]

look_for_duplicates (np.concatenate((train_dataset,valid_dataset,test_dataset)))




