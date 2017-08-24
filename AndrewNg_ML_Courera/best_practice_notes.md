# Feature scaling and mean normalization
https://www.coursera.org/learn/machine-learning/supplement/CTA0D/gradient-descent-in-practice-i-feature-scaling
```
xi:= (xi−μi)/si
```
In pracitce, the range within ```[-3, 3]``` and outside ```[-1/3,1/3]``` is good.

# Choosing gradient descent step size (alpha)

try step sizes in factors of 3 and choose the one immediately smaller than the largest working step size.

# Normal Equation vs Gradient Descent

With the normal equation, computing the inversion has complexity `O(n^3)`. So if we have a very large number of features, the normal equation will be slow. In practice, when n exceeds 10,000 it might be a good time to go from a normal solution to an iterative process.

# Dataset

A typical split of a dataset into training, validation and test sets might be 
* 60% training set, 
* 20% validation set, and 
* 20% test set.
