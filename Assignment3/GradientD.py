from __future__ import division
import numpy as np
import computeCost

def gradientD(X, y, theta, alpha, num_iters):
    m = len(y)
    J_history = np.zeros(num_iters)
    for iter in range(num_iters):
        theta = theta - (alpha/m)*np.dot(X.T,(X*theta - y))
        J_history[iter] = computeCost.computeCost(X, y, theta)

    return [theta, J_history]

def gradientDSoftMax(X, y, theta, alpha, num_iters):
    m = len(y)
    J_history = np.zeros(num_iters)
    for iter in range(num_iters):
        theta = theta - (alpha/m)*np.dot(X.T,(X*theta - y))
        J_history[iter] = computeCost.computeCostSoftMax(X, y, theta)

    return [theta, J_history]
