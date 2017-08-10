function p = predict(Theta1, Theta2, X)
%PREDICT Predict the label of an input given a trained neural network
%   p = PREDICT(Theta1, Theta2, X) outputs the predicted label of X given the
%   trained weights of a neural network (Theta1, Theta2)

% Useful values
m = size(X, 1);
num_labels = size(Theta2, 1); %num_labels=10

% You need to return the following variables correctly 
p = zeros(size(X, 1), 1); 

% ====================== YOUR CODE HERE ======================
% Instructions: Complete the following code to make predictions using
%               your learned neural network. You should set p to a 
%               vector containing labels between 1 to num_labels.
%
% Hint: The max function might come in useful. In particular, the max
%       function can also return the index of the max element, for more
%       information see 'help max'. If your examples are in rows, then, you
%       can use max(A, [], 2) to obtain the max for each row.
%

% Theta1 has size 25 x 401
% Theta2 has size 10 x 26

% Add ones to the X data matrix
A1 = [ones(m, 1) X]; % A: m x (n+1) where n is the number of pixels of image (400 in this case)
Z2 = A1 * Theta1'; % Z2: m x 25
A2 = sigmoid(Z2);
A2 = [ones(m, 1) A2]; % A2: m x 26
Z3 = A2 * Theta2'; % Z3: m x 10
A3 = sigmoid(Z3);
[maxVals, p]=max(A3, [], 2);



% =========================================================================


end
