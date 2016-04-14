from sklearn.datasets import fetch_mldata
import numpy as np
import os
from pylab import *
from GradientD import *
from computeCost import sigmoid

mnist = fetch_mldata('MNIST original') # get the data
mnist.data.shape
mnist.target.shape
np.unique(mnist.target)

X, y = mnist.data / 255., mnist.target
X_train, X_test = X[:60000], X[60000:]
y_train, y_test = y[:60000], y[60000:]
X_train.shape
y_train.shape
size=len(y_train)

## extract "3" digits and show their average"
ind = [ k for k in range(size) if y_train[k]==4 ]
extracted_images=X_train[ind,:]

mean_image=extracted_images.mean(axis=0)
imshow(mean_image.reshape(28,28), cmap=cm.gray)
show()

from sklearn import datasets
iris = datasets.load_iris()

# Q1:

# logistic regression writed by sklearn
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression()
classifier.fit(iris.data, iris.target)
y_pred = classifier.predict(iris.data)
print('The numbers of wrong: %d' % (iris.target != y_pred).sum())

# logistic regression writed by me
m = iris.data[0:100,]
n = iris.target[0:100,]
n = array([n]).T
k = np.ones((n.size,1))
m = np.hstack([k,m])
theta = np.dot(np.dot(np.linalg.inv(np.dot(m.T,m)),m.T),n)
print(theta)
iterations = 1500
alpha = 0.01
theta = np.matrix(np.zeros(m.shape[1])).T #initialize fitting parameters
[theta, J_history] = gradientD(m,n,theta,alpha,iterations)
n_pred = sigmoid(m*theta)
for i in range(len(n_pred)):
    if(n_pred[i]>=0.5):
        n_pred[i] = 1
    else:
        n_pred[i] = 0
print('The numbers of wrong: %d' % ( n != n_pred).sum())


# Q2:

classifier = LogisticRegression()
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)
print('The numbers of wrong: %d' % (y_test != y_pred).sum())

# Q3:

# Multilayer perceptron by scikit-learn
X_train3 = X_train[:12664]
y_train3 = y_train[:12664]
X_test3 = X_test[:2114]
y_test3 = y_test[:2114]

from sklearn.linear_model import Perceptron
ppn = Perceptron(n_iter=40,eta0=0.01,random_state=0)
ppn.fit(X_train3,y_train3)

y_pred = ppn.predict(X_test3)
print('The numbers of wrong: %d' % (y_test3 != y_pred).sum())

# Q4:

# Multilayer perceptron by scikit-learn

from sklearn.linear_model import Perceptron
ppn = Perceptron(n_iter=40,eta0=0.01,random_state=0)
ppn.fit(iris.data,iris.target)

y_pred = ppn.predict(iris.data)
print('The numbers of wrong: %d' % (iris.target != y_pred).sum())