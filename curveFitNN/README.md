# Problem Description
Fit a curve.

## Training Data

100 random data points generated from y=f(x) on [-5,5].

## Neural Network

* Input Node: 1
* Hidden Nodes: 3 (can be varied)
* Output Node: 1

### Activation Function

### Regularization?

## Test Data

10000 equally space data points generated from y=f(x) on [-5, 5].

# Code

* [curvefit.py](./curvefit.py): old style code based on [basicClassifierNN](../basicClassifierNN) (and of course its reference blog)
* [tf_fit.py](./tf_fit.py): first attempt to use TensorFlow in a simple rough way
* [tf_fit2.py](./tf_fit2.py): more of less following the [MNIST Tutorial](https://www.tensorflow.org/versions/r0.10/tutorials/mnist/beginners/index.html)
