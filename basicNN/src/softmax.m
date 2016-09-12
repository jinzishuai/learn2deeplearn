#ref: https://en.wikipedia.org/wiki/Softmax_function
function y = softmax(x)
    expX=exp(x);
    width=size(x,2);
    sumExpX=sum(expX,2);
    y = expX./repmat(sumExpX,1,width);
end
