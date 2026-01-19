# Mathematical Optimization - Homework 2'
The written part of the exercises can be found [here](https://bwsyncandshare.kit.edu/s/N7DE45Pd7rppEBb). The notes on exercise 4 are due to Louka Moroni.

The coded parts can be found as the following:
 - exercise 2.1: file `gradient_descent.py` in function `prox_transform`
 - exercise 3: files `lasso_recovery.py` and `gradient_descent.py`
 - exercise 5: files `linear_classifier.py`, `exercise1.py` and `exercise2.py`

The MATLAB code in `faces` has been transpiled to python, and the transpilation has been tested.

The Lasso recovery works quite decently, getting an absolute error of less than 10 for almost all input dimensions and ranges, often below 1.

The linear classifier pipeline has been fully set up and functions, however, the Logistic loss function and thus problems (M1) and (M2) don't produce stable results even after extensive grid search on the parameters.

The mean average image hyperplane classifier in exercise 5 roughly classifies half of the images correctly. Of course, it can not compensate for e.g. the background being different in different images, so no meaningful classification is possible. It is as good as guessing randomly.
