from __future__ import print_function
import matplotlib.pyplot as plt
import numpy as np
from sklearn import  linear_model
import data

letters=np.array(['A', 'B', 'C','D','E','F','G','H','I','J'])
train_dataset, train_labels, valid_dataset, valid_labels, test_dataset, test_labels = data.load_all_data_from_single_pickle_file('../../notMNIST.pickle')

for n in [50, 100, 1000, 5000]:
    testrange=range(n)
    logistic = linear_model.LogisticRegression()
    print('%d training samples: LogisticRegression score: %f'
          % (n, logistic.fit(np.reshape(train_dataset[testrange],(-1,28*28)), train_labels[testrange]).score(np.reshape(test_dataset, (-1, 28*28)), test_labels)))