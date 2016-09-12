function data_loss=calculate_loss(X, W1, W2, b1, b2,yy,reg_lambda)
#Calculate the loss
    num_examples=length(X);
    
    # Forward propagation
    z1 = X*W1 + repmat(b1, num_examples,1); #we use repmat to apply the constant to all input values, dimension num_examples x nn_hdim
    a1 = tanh(z1); #dimension num_examples x nn_hdim
    z2 = a1*W2 + repmat(b2, num_examples,1); #dimension num_examples x nn_output_dim
    probs = softmax(z2);  #dimension num_examples x nn_output_dim
    
    #Question: how does this work here to calculate the cross entropy loss
    #This one is tricky
    #Python code: corect_logprobs = -np.log(probs[range(num_examples), y]), which takes the indexes of the values y
    #It takes the sub array out of probs array
    #We implement it in octave in a different way
    corect_logprobs =  - log( sum(probs.*yy,2) );
    data_loss = sum(corect_logprobs); #scalar
    data_loss += reg_lambda/2 * ( sum(sum(W1.^2)) + sum(sum(W2.^2)) );
    data_loss /= num_examples;
end
