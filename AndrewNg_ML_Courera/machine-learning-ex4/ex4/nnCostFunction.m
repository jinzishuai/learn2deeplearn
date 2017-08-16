function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0; % Scalar
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%


% Theta1: hidden_layer_size x (input_layer_size + 1), eg, 25 x 401
% Theta2: num_labels x (hidden_layer_size + 1), eg, 10 x 26
% X: 5000 x 400
% y: 5000 x 1


% As mentioned in lecture 9, we should loop through all sample data. 
% In fact, this part can be done with a Matrix implementation
% But as recommended by Ng, we start with a loop
% The Maxtri implemetation is similar to the predict.m in ex3
Cost = zeros(m, 1); % 5000 x 1
DELTA1=zeros(size(Theta1));
DELTA2=zeros(size(Theta2));
for i = 1:m 
  a1 = [1; X(i, :)']; %401 x 1
  z2 = Theta1 * a1; % 25 x 1
  a2 = [1;sigmoid(z2)]; % 26 x 1
  z3 = Theta2 * a2; % 10 x 1
  a3 = sigmoid(z3); % 10 x 1
  h = a3; % 10 x 1
  
  % vectorize the sample output
  yVeci = zeros(num_labels, 1); % 10 x 1
  yVeci(y(i)) = 1;
  
  Cost(i) = - yVeci'*log(h) - (1-yVeci')*log(1-h); 
 
  delta3 = a3 -yVeci; % 10 x 1
  delta2 = (Theta2' * delta3).*sigmoidGradient([0;z2]); %26 x 1, computes delta2(1) anyway, just don't use it
  DELTA2 = DELTA2 + delta3*a2'; % 10 x 26
  DELTA1 = DELTA1 + delta2(2:hidden_layer_size+1)*a1'; %25 x 401
  
  
end
J = mean(Cost);
Theta1_grad=DELTA1/m;
Theta2_grad=DELTA2/m;

% Now Add regulization, skipping the bias terms
Theta1(:,1)=0; 
Theta2(:,1)=0;
J= J + lambda/(2*m)*(sum(Theta1(:).^2)+sum(Theta2(:).^2));

GradientRegulization1= lambda/m*Theta1;
GradientRegulization2= lambda/m*Theta2;

Theta1_grad=Theta1_grad+GradientRegulization1;
Theta2_grad=Theta2_grad+GradientRegulization2;







% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
