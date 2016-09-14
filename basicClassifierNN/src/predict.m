clear;
load final.mat
num_x=200;
num_y=100;
num_examples=num_x*num_y;
x=linspace(-2,2.5,num_x);
y=linspace(-1,1.5,num_y);
[xx,yy]=meshgrid(x,y);
X=[reshape(xx,num_examples,1),reshape(yy,num_examples,1)];
# Forward propagation
z1 = X*W1 + repmat(b1, num_examples,1); #we use repmat to apply the constant to all input values, dimension num_examples x nn_hdim
a1 = tanh(z1); #dimension num_examples x nn_hdim
z2 = a1*W2 + repmat(b2, num_examples,1); #dimension num_examples x nn_output_dim
probs = softmax(z2);  #dimension num_examples x nn_output_dim
[argvalue, argmax] = max(probs,[],2);
Y=argmax-1; #argmax is either 1 or 2 while Y is 0 or 1
C=reshape(Y,size(xx));
contourf(xx,yy,C);
hold on;
#Plot training data
result=load('../dataset/result.dat');
scatter(result(:,1),result(:,2),[],result(:,3)+2,"filled");

