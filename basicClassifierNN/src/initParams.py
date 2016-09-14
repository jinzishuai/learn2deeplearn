#!/usr/bin/python
import numpy as np
from sklearn import datasets, linear_model
import matplotlib.pyplot as plt
#Define constants
nn_input_dim = 2; # input layer dimensionality
nn_output_dim = 2; # output layer dimensionality
nn_hdim = 3;
num_passes = 2000; #20000;

#Initialize parameters to random numbers
np.random.seed(0)
W1=np.random.randn(nn_input_dim, nn_hdim)/np.sqrt(nn_input_dim); #dimension nn_input_dim x nn_hdim
W2=np.random.randn(nn_hdim, nn_output_dim) /np.sqrt(nn_hdim); #dimension nn_hdim x nn_output_dim
np.savetxt('initW1.dat', W1,fmt="%.20f %.20f %.20f", delimiter='\n')
np.savetxt('initW2.dat', W2,fmt="%.20f %.20f", delimiter='\n')

