# Problem Description

This is the same problem described by [this reference blog](http://www.wildml.com/2015/09/implementing-a-neural-network-from-scratch/).

**Objective**: train a Machine Learning **classifier** that predicts the correct class (male (1) or female (0)) given the x- and y- coordinates.

# Training Data

The same as in the reference and can be generated by the [dataGen.py](dataset/dataGen.py) script.

![image](http://d3kbpzbmcynnmx.cloudfront.net/wp-content/uploads/2015/09/nn-from-scratch-dataset.png)

# Neural Network

* Input=(x,y) 2D coordinates of points
* Output=(probability of male, probability of female) vector of probabilities whose sum is 1. It can also be described by a single integer as the index of the higher probability (0 for male as for the first index, 1 for female as for the second index)
  * (1,0) => 0 => male
  * (0,1) => 1 => female

## Prediction Result

![image](src/prediction.png)

## Code

It is almost an exact rewrite on the original python code with Matlab/octave since I am more familiar with that.

# References

* http://www.wildml.com/2015/09/implementing-a-neural-network-from-scratch/
* https://mattmazur.com/2015/03/17/a-step-by-step-backpropagation-example/

## Background Knowledge

### Logistic Regression/Classifier

* https://en.wikipedia.org/wiki/Logistic_regression
* https://en.wikipedia.org/wiki/Gradient_descent
* https://en.wikipedia.org/wiki/Regularization_(mathematics)
