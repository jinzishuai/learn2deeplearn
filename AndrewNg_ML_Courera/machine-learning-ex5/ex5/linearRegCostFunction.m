function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
%   cost of using theta as the parameter for linear regression to fit the 
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost and gradient of regularized linear 
%               regression for a particular choice of theta.
%
%               You should set J to the cost and grad to the gradient.
%



% X: m x (n+1) where m is number of examples and n is the number of features
% h, y: m x 1
% theta: (n+1) x 1: theta(1) is the bias
% grad: (n+1) x 1
n = length(theta) -1; 
h= X* theta;
theta(1)=0; %avoid regularize on the bias team of theta(1)
J=1/(2*m)*(sumsq(h-y)+lambda*sumsq(theta));
grad=1/m*((X'*(h-y))+lambda*theta);





% =========================================================================

grad = grad(:);

end
