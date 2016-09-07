clear;
clc;
#Load the data
result=load('../dataset/result.dat');
X=result(:,1:2);
y=result(:,3); #Scalar form of the output
yy=[1-y,y]; #Vector form of the output

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
W1=randn(nn_input_dim, nn_hdim)/ sqrt(nn_input_dim); #dimension nn_input_dim x nn_hdim
b1=zeros(1, nn_hdim);
W2=randn(nn_hdim, nn_output_dim) /sqrt(nn_hdim); #dimension nn_hdim x nn_output_dim
b2=zeros(1,nn_output_dim);

for i=0:num_passes
    # Forward propagation
    z1 = X*W1 + repmat(b1, num_examples,1); #we use repmat to apply the constant to all input values, dimension num_examples x nn_hdim
    a1 = tanh(z1); #dimension num_examples x nn_hdim
    z2 = a1*W2 + repmat(b2, num_examples,1); #dimension num_examples x nn_output_dim
    probs = softmax(z2);  #dimension num_examples x nn_output_dim
    
    # Backward progagation
    delta3 = probs - yy; #dimension num_examples x nn_output_dim 
    dW2 = a1'*delta3; #same dimension of W2: nn_hdim x nn_output_dim
    
end 

