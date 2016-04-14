from __future__ import division
import numpy as np

# Compute cost for linear regression
def computeCost(X, y, theta):
    m = len(y)  #number of tranning examples
    J = 0
    J = 1/(2*m)*np.dot((sigmoid(X*theta) - y).T,(sigmoid(X*theta) - y))
    return J

def computeCostSoftMax(X, y, theta):
    m = len(y)  #number of tranning examples
    J = 0
    J = 1/(2*m)*np.dot((softMaxFunction(X*theta) - y).T,(softMaxFunction(X*theta) - y))
    return J

def sigmoid(inX):
    return 1.0/(1+np.exp(-inX))

def softMaxFunction(weight, X, j):
    return float(np.exp(weight[j].dot(X))) / np.sum(np.exp(weight.dot(X)), axis=0)