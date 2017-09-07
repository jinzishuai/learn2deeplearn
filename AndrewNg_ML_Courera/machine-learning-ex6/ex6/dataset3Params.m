function [C, sigma] = dataset3Params(X, y, Xval, yval)
%DATASET3PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = DATASET3PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly
C = 1;
sigma = 0.1;

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using 
%        mean(double(predictions ~= yval))
%

%CRange=[0.01 0.03 0.1 0.3 1 3 10 30];
%%CRange=[0.01 0.03]
%CCount=length(CRange);
%sigmaRange=[0.01 0.03 0.1 0.3 1 3 10 30];
%sigmaCount=length(sigmaRange);
%preErr=zeros(CCount*sigmaCount,1);
%for i=1:CCount
%  for j=1:sigmaCount
%    C=CRange(i);
%    sigma=sigmaRange(j);
%    model= svmTrain(X, y, C, @(x1, x2) gaussianKernel(x1, x2, sigma));
%    predictions = svmPredict(model, Xval);
%    preErr(sub2ind([CCount,sigmaCount],i,j))=mean(double(predictions ~= yval));
%  end
%end
%preErr
%[minErr,idx]=min(preErr);
%[CIdx,sigmaIdx]=ind2sub([CCount,sigmaCount],idx);
%C=CRange(CIdx);
%sigma=sigmaRange(sigmaIdx);
%fprintf("the optimal parameters are C=%f, sigma=%f while the prediction error=%f",C, sigma, minErr);
% =========================================================================

end
