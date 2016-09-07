#ref: https://en.wikipedia.org/wiki/Softmax_function
function y = softmax(x)
    expX=exp(x);
    len=length(x);
    sumExpX=sum(expX,1);
    y = expX./repmat(sumExpX,len,1);
end
