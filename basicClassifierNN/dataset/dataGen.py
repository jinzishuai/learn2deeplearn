#!/usr/bin/python
import numpy as np
from sklearn import datasets, linear_model
import matplotlib.pyplot as plt

# Generate a dataset and plot it
np.random.seed(0) #this data will be predictable, ie, always the same
X, y = datasets.make_moons(200, noise=0.20)
result = np.concatenate((X,y[:,None]),axis=1)
np.savetxt('result.dat', result, fmt='%.20f %.20f %d', delimiter='\n')

#visualize
plt.scatter(X[:,0], X[:,1], s=40, c=y, cmap=plt.cm.Spectral)
plt.show()
