clear;
clc;
#Load the data
result=load('../dataset/result.dat');
X=result(:,1:2);

#Define constants
num_examples = length(X); # training set size
nn_input_dim = 2; # input layer dimensionality
nn_output_dim = 2; # output layer dimensionality
 
# Gradient descent parameters (I picked these by hand)
epsilon = 0.01; # learning rate for gradient descent
reg_lambda = 0.01; # regularization strength

#Parameters to tweak
nn_hdim = 3;
num_passes = 1; #20000;

#Initialize parameters to random numbers
W1=randn(nn_input_dim, nn_hdim)/ sqrt(nn_input_dim);
b1=zeros(1, nn_hdim);
W2=randn(nn_hdim, nn_output_dim) /sqrt(nn_hdim);
b2=zeros(1,nn_output_dim);

for i=0:num_passes
    # Forward propagation
    z1 = X*W1 + repmat(b1, num_examples,1); #we use repmat to apply the constant to all input values
    a1 = tanh(z1);
    z2 = a1*W2 + repmat(b2, num_examples,1);
    a2 = softmax(z2);
    
end 

