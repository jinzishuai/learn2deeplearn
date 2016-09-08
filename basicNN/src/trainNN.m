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
num_passes = 2000; #20000;

#Initialize parameters to random numbers
W1=randn(nn_input_dim, nn_hdim)/ sqrt(nn_input_dim); #dimension nn_input_dim x nn_hdim
b1=zeros(1, nn_hdim); #dimension 1 x nn_hdim
W2=randn(nn_hdim, nn_output_dim) /sqrt(nn_hdim); #dimension nn_hdim x nn_output_dim
b2=zeros(1, nn_output_dim); #dimension 1 x nn_output_dim

for i=0:num_passes
    # Forward propagation
    z1 = X*W1 + repmat(b1, num_examples,1); #we use repmat to apply the constant to all input values, dimension num_examples x nn_hdim
    a1 = tanh(z1); #dimension num_examples x nn_hdim
    z2 = a1*W2 + repmat(b2, num_examples,1); #dimension num_examples x nn_output_dim
    probs = softmax(z2);  #dimension num_examples x nn_output_dim
    #Calculate the loss
    #Question: how does this work here to calculate the cross entropy loss
    corect_logprobs =  - log( probs);
    data_loss = sum(sum(corect_logprobs)); #scalar
    data_loss += reg_lambda/2 * ( sum(sum(W1.^2)) + sum(sum(W2.^2)) );
    data_loss /= num_examples;
    if mod(i,1000) == 0
        printf("iteration %d, data_loss=%f\n",i, data_loss);
    end
    
    
    # Backward progagation
    # Question: how does the delta3 calculation translate into the Python code of ```delta3[range(num_examples), y] -= 1```?
    delta3 = probs - yy; #dimension num_examples x nn_output_dim
    dW2 = a1'*delta3; #same dimension of W2: nn_hdim x nn_output_dim
    db2 = sum(delta3); 
    #Question: But why do we do that sum? Should we average it by dividing by num_examples?
    #Answer: the regularization function contains of sum of all testing data
    delta2 = (delta3* W2').* (1- a1.^2); # dimension: num_examples x nn_hdim
    dW1 = X'*delta2;
    db1 = sum(delta2); #Question: same as in db2

    # Add regularization terms (b1 and b2 don't have regularization terms)
    dW2 += reg_lambda * W2;
    dW1 += reg_lambda * W1;

    # Gradient descent parameter update
    W1 += -epsilon * dW1;
    b1 += -epsilon * db1;
    W2 += -epsilon * dW2;
    b2 += -epsilon * db2;
end 

