#!/usr/bin/python
import random
import pylab as pl
import numpy as np

def makeTerrainData(n_points=1000):
    ###############################################################################
    ### make the toy dataset##########################
    random.seed(42)
    grade = [random.random() for ii in range(0,n_points)]
    bumpy = [random.random() for ii in range(0,n_points)]
    error = [random.random() for ii in range(0,n_points)]
    y = [round(grade[ii]*bumpy[ii]+0.3+0.1*error[ii]) for ii in range(0,n_points)]
    for ii in range(0, len(y)):
        if grade[ii]>0.8 or bumpy[ii]>0.8:
            y[ii] = 1.0

        ### split into train/test sets
    X = [[gg, ss] for gg, ss in zip(grade, bumpy)]
    split = int(0.75*n_points)
    X_train = X[0:split]
    X_test  = X[split:]
    y_train = y[0:split]
    y_test  = y[split:]

    grade_sig = [X_train[ii][0] for ii in range(0, len(X_train)) if y_train[ii]==0]
    bumpy_sig = [X_train[ii][1] for ii in range(0, len(X_train)) if y_train[ii]==0]
    grade_bkg = [X_train[ii][0] for ii in range(0, len(X_train)) if y_train[ii]==1]
    bumpy_bkg = [X_train[ii][1] for ii in range(0, len(X_train)) if y_train[ii]==1]

    training_data = {"fast":{"grade":grade_sig, "bumpiness":bumpy_sig}
        , "slow":{"grade":grade_bkg, "bumpiness":bumpy_bkg}}


    grade_sig = [X_test[ii][0] for ii in range(0, len(X_test)) if y_test[ii]==0]
    bumpy_sig = [X_test[ii][1] for ii in range(0, len(X_test)) if y_test[ii]==0]
    grade_bkg = [X_test[ii][0] for ii in range(0, len(X_test)) if y_test[ii]==1]
    bumpy_bkg = [X_test[ii][1] for ii in range(0, len(X_test)) if y_test[ii]==1]

    test_data = {"fast":{"grade":grade_sig, "bumpiness":bumpy_sig}
        , "slow":{"grade":grade_bkg, "bumpiness":bumpy_bkg}}

    return X_train, y_train, X_test, y_test



def gen_lin_separable_data():
    # generate training data in the 2-d case
    mean_1 = np.array([0, 2])
    mean_2 = np.array([2, 0])
    
    cov = np.array([[0.8, 0.6], [0.6, 0.8]])
    
    X_1 = np.random.multivariate_normal(mean1, cov, 2000)
    y_1 = np.ones(len(X_1))
    X_2 = np.random.multivariate_normal(mean2, cov, 2000)
    y_2 = np.ones(len(X_2)) * -1
    
    return X1, y1, X2, y2

def gen_non_lin_separable_data():
    mean_1 = [-1, 2]
    mean_2 = [1, -1]
    mean_3 = [4, -4]
    mean_4 = [-4, 4]
    
    cov = [[1.0,0.8], [0.8, 1.0]]
    
    X_1 = np.random.multivariate_normal(mean_1, cov, 2000)
    X_1 = np.vstack((X_1, np.random.multivariate_normal(mean_3, cov, 2000)))
    y_1 = np.ones(len(X_1))
    X_2 = np.random.multivariate_normal(mean_2, cov, 2000)
    X_2 = np.vstack((X_2, np.random.multivariate_normal(mean_4, cov, 2000)))
    y_2 = np.ones(len(X_2)) * -1
    
    return X1, y1, X2, y2

def gen_lin_separable_overlap_data():
    # generate training data in the 2-d case
    mean_1 = np.array([0, 2])
    mean_2 = np.array([2, 0])
    
    cov = np.array([[1.5, 1.0], [1.0, 1.5]])
    
    X_1 = np.random.multivariate_normal(mean_1, cov, 2000)
    y_1 = np.ones(len(X_1))
    X_2 = np.random.multivariate_normal(mean_2, cov, 2000)
    y_2 = np.ones(len(X_2)) * -1
    
    print X_1,y_1,X_2,y_2
    return X_1, y_1, X_2, y_2

def split_train(X_1, y_1, X_2, y_2):
    X1_train = X_1[:90]
    y1_train = y_1[:90]
    X2_train = X_2[:90]
    y2_train = y_2[:90]
    
    X_train = np.vstack((X1_train, X2_train))
    y_train = np.hstack((y1_train, y2_train))
    
    return X_train, y_train

def split_test(X_1, y_1, X_2, y_2):
    
    X1_test = X_1[90:]
    y1_test = y_1[90:]
    X2_test = X_2[90:]
    y2_test = y_2[90:]
    
    X_test = np.vstack((X1_test, X2_test))
    y_test = np.hstack((y1_test, y2_test))
    
    return X_test, y_test

def plot_margin(X1_train, X2_train, clf):
    def f(x, w, b, c=0):
        # given x, return y such that [x,y] in on the line
        # w.x + b = c
        return (-w[0] * x - b + c) / w[1]

    pl.plot(X1_train[:,0], X1_train[:,1], "ro")
    pl.plot(X2_train[:,0], X2_train[:,1], "bo")
    pl.scatter(clf.sv[:,0], clf.sv[:,1], s=100, c="g")

    # w.x + b = 0
    a0 = -4; a1 = f(a0, clf.w, clf.b)
    b0 = 4; b1 = f(b0, clf.w, clf.b)
    pl.plot([a0,b0], [a1,b1], "k")

    # w.x + b = 1
    a0 = -4; a1 = f(a0, clf.w, clf.b, 1)
    b0 = 4; b1 = f(b0, clf.w, clf.b, 1)
    pl.plot([a0,b0], [a1,b1], "k--")

    # w.x + b = -1
    a0 = -4; a1 = f(a0, clf.w, clf.b, -1)
    b0 = 4; b1 = f(b0, clf.w, clf.b, -1)
    pl.plot([a0,b0], [a1,b1], "k--")

    pl.axis("tight")
    pl.show()

def plot_contour(X1_train, X2_train, clf):
    pl.plot(X1_train[:,0], X1_train[:,1], "ro")
    pl.plot(X2_train[:,0], X2_train[:,1], "bo")
    pl.scatter(clf.sv[:,0], clf.sv[:,1], s=100, c="g")

    X1, X2 = np.meshgrid(np.linspace(-6,6,50), np.linspace(-6,6,50))
    X = np.array([[x1, x2] for x1, x2 in zip(np.ravel(X1), np.ravel(X2))])
    Z = clf.project(X).reshape(X1.shape)
    
    pl.contour(X1, X2, Z, [0.0], colors='k', linewidths=1, origin='lower')
    pl.contour(X1, X2, Z + 1, [0.0], colors='grey', linewidths=1, origin='lower')
    pl.contour(X1, X2, Z - 1, [0.0], colors='grey', linewidths=1, origin='lower')

    pl.axis("tight")
    pl.show()