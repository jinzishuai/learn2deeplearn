function [J, grad] = cofiCostFunc(params, Y, R, num_users, num_movies, ...
                                  num_features, lambda)
%COFICOSTFUNC Collaborative filtering cost function
%   [J, grad] = COFICOSTFUNC(params, Y, R, num_users, num_movies, ...
%   num_features, lambda) returns the cost and gradient for the
%   collaborative filtering problem.
%

% Unfold the U and W matrices from params
X = reshape(params(1:num_movies*num_features), num_movies, num_features);
Theta = reshape(params(num_movies*num_features+1:end), ...
                num_users, num_features);

            
% You need to return the following values correctly
J = 0;
X_grad = zeros(size(X));
Theta_grad = zeros(size(Theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost function and gradient for collaborative
%               filtering. Concretely, you should first implement the cost
%               function (without regularization) and make sure it is
%               matches our costs. After that, you should implement the 
%               gradient and use the checkCostFunction routine to check
%               that the gradient is correct. Finally, you should implement
%               regularization.
%
% Notes: X - num_movies  x num_features matrix of movie features
%        Theta - num_users  x num_features matrix of user features
%        Y - num_movies x num_users matrix of user ratings of movies
%        R - num_movies x num_users matrix, where R(i, j) = 1 if the 
%            i-th movie was rated by the j-th user
%
% You should set the following variables correctly:
%
%        X_grad - num_movies x num_features matrix, containing the 
%                 partial derivatives w.r.t. to each element of X
%        Theta_grad - num_users x num_features matrix, containing the 
%                     partial derivatives w.r.t. to each element of Theta
%

diffMat=(X*Theta'-Y).^2; % num_movies x num_users
J=sum(sum(R.*diffMat));
J=J/2;

% Gradients
for i = 1:num_movies
	idx=find(R(i,:)==1); % 1 x num_users_rates_movie_i
	Theta_temp=Theta(idx,:); % num_users_rates_movie_i x num_features
	Y_temp=Y(i,idx); % 1 x num_users_rates_movie_i
	X_grad(i,:)=(X(i,:)*Theta_temp'-Y_temp)*Theta_temp;  % 1 x num_features
end
for j = 1:num_users
	idx=find(R(:,j)==1); % 1 x num_movies_users_j_rated
	Theta_temp=Theta(j,:); %  1 x num_features
	Y_temp=Y(idx,j); % num_movies_users_j_rated x 1
	%X(idx,:): num_movies_users_j_rated x num_features
	Theta_grad(j,:)=(X(idx,:)*Theta_temp'-Y_temp)'* X(idx,:); % 1 x num_features 
end


%Add regulization
J=J+lambda/2*(sum(sum(Theta.^2))+sum(sum(X.^2)));
X_grad=X_grad+lambda*X;
Theta_grad=Theta_grad+lambda*Theta;







% =============================================================

grad = [X_grad(:); Theta_grad(:)];

end
