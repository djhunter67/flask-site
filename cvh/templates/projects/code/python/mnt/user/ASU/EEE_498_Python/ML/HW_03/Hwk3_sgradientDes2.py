# -*- coding: utf-8 -*-
"""
Created on Thu May 21 15:16:36 2020

@author: olhartin@asu.edu
"""
# conda install -c omnia autograd
from re import L
from autograd import grad
from autograd import numpy as np
import matplotlib.pyplot as plt
from time import process_time
from scipy.linalg import norm


z = (1+2.236)/2  # Golden ratio
tol = 1e-6
maxiter = 2000
s = 1
all = 1
g0orag1 = 0


def main():

    ##
    # initial solution
    ##
    w0 = np.transpose(np.array([-0.3, 0.2]))
    ModelResults = []
    ##
    # plot errors versus iteration
    ##

    ##
    # constant learning rate case
    ##
    if (all):
        print('\n 2b) constant learning rate\n')
        w = w0
        alpha = 0.1
        w, count, error, dtime = grad_des(f, w, alpha, maxiter, tol, g0orag1=0)
        if count >= maxiter:
            print('Didnt Converge')
        else:
            print('Converged')
        ModelResult = {'Analytical Diff': g0orag1, 'Method': 'constant learning rate', 'lr': alpha,
                       'Count': count, 'Beta': 0, 'Weight': w, 'Error': error, 'dtime': dtime}
        ModelResults.append(ModelResult)

        # Learning rate drops w/ iteration
        print('\n 3) learning rate drops w/ iteration\n')
        w = w0
        alpha = 1
        w, count, error, dtime = grad_desk(f, w, alpha, maxiter, tol, g0orag1=0)
        if count >= maxiter:
            print('Didnt Converge')
        else:
            print('Converged')
        ModelResult = {'Analytical Diff': g0orag1, 'Method': 'learning rate declines', 'lr': alpha,
                       'Count': count, 'Beta': 0, 'Weight': w, 'Error': error, 'dtime': dtime}
        ModelResults.append(ModelResult)

        # Normalized
        print('\n 4) Normalized \n')
        w = w0
        alpha = 0.001
        w, count, error, dtime = grad_desn(f, w, alpha, maxiter, tol, g0orag1=0)
        if count >= maxiter:
            print('Didnt Converge')
        else:
            print('Converged')
        ModelResult = {'Analytical Diff': g0orag1, 'Method': 'Normalized', 'lr': alpha,
                       'Count': count, 'Beta': 0, 'Weight': w, 'Error': error, 'dtime': dtime}
        ModelResults.append(ModelResult)

        # Lipshitz
        print('\n 5) Lipshitz \n')
        w = w0
        alpha = 2
        w, count, error, dtime = grad_desL(f, w, alpha, maxiter, tol, g0orag1=0)
        if count >= maxiter:
            print('Didnt Converge')
        else:
            print('Converged')
        ModelResult = {'Analytical Diff': g0orag1, 'Method': 'Gradient Descents Lipshitz', 'lr': alpha,
                       'Count': count, 'Beta': 0, 'Weight': w, 'Error': error, 'dtime': dtime}
        ModelResults.append(ModelResult)

        # Steepest Descent
        print('\n 6) No Momentum and Steepest Descent \n')
        w = w0
        alpha = 0.01
        w, count, error, dtime = grad_desm(f, w, alpha, 0, maxiter, tol, s, g0orag1=0)
        if count >= maxiter:
            print('Didnt Converge')
        else:
            print('Converged')
        ModelResult = {'Analytical Diff': g0orag1, 'Method': ' no momentum, steepest grad', 'lr': alpha,
                       'Count': count, 'Beta': 0, 'Weight': w, 'Error': error, 'dtime': dtime}
        ModelResults.append(ModelResult)

        # Momentum & Steepest Descent
        print('\n 7) Momentum and Steepest Descent \n')
        w = w0
        alpha = 0.01
        w, count, error, dtime = grad_desq(f, w, alpha, 0, maxiter, tol, s, g0orag1=0)
        if count >= maxiter:
            print('Didnt Converge')
        else:
            print('Converged')
        ModelResult = {'Analytical Diff': g0orag1, 'Method': ' momentum, steepest grad', 'lr': alpha,
                       'Count': count, 'Beta': 0, 'Weight': w, 'Error': error, 'dtime': dtime}
        ModelResults.append(ModelResult)



def ploterrors(errors):
    plt.plot(np.log10(errors))
    plt.grid()
    plt.xlabel('iteration')
    plt.ylabel('error')
    plt.show()
    return 0
##
# Search golden number
##


def goldendelta(x4, x1, z):
    return((x4-x1)/z)
##
# golden search
##


def goldensearch(g, w, h, x1, x4, accuracy):
    # initial positions of the four points
    ##  x1 = sigma/10
    ##  x4 = sigma*10
    x2 = x4 - goldendelta(x4, x1, z)
    x3 = x1 + goldendelta(x4, x1, z)

# initial values of the function at the four points
    f1 = g(w-x1*h)
    f2 = g(w-x2*h)
    f3 = g(w-x3*h)
    f4 = g(w-x4*h)
    i = 0
    error = abs(x4-x1)
    while error > accuracy:
        if (f2 < f3):
            x4, f4 = x3, f3
            x3, f3 = x2, f2
            x2 = x4 - goldendelta(x4, x1, z)
            f2 = g(w-x2*h)
        else:
            x1, f1 = x2, f2
            x2, f2 = x3, f3
            x3 = x1 + goldendelta(x4, x1, z)
            f3 = g(w-x3*h)
        i += 1
        error = abs(x4-x1)
    return((x1+x4)/2.0, i, error)
##
# golden search driver
##


def golden(g, w, h, alpha):
    alpha, iter, error = goldensearch(g, w, h, alpha/10., alpha*10.0, 1e-6)
    return alpha
##
# magnitude of a vector
##


def magnitude(x):
    return(np.linalg.norm(x))


##
# function constants
##
a = 20.
b = np.array([1., -1.]).reshape(2, 1)
C = np.array([[1., 0], [0., 2.]])
D = np.array([[1., -2], [1., 2.]])
##
# base function, can be quadriatic (convex) or not
##

######################################################
# NUMBER 1
######################################################
# Part a


def f(w):
    w = w.T
    w = w[0]
    b = np.array([1., -1.])
    return a + np.dot(b, w) + np.dot(w, np.dot(C, w)) + np.dot(w, np.dot(w, np.dot(D, w), w))
##
# First derivative
##

# Part b


def df(w):
    return b + 2.0 * np.matmul(C, w)\
        + 2. * np.matmul(w.T, np.matmul((D + D.T), w))[0][0] * w
##
# Second derivative
##


def ddf(w):
    return C + 12 * np.dot(w.T, np.dot(D + D.T, w))
##
# absolute value of function, nonlinear
##


def fabs(w):
    return(np.abs(f(w)))
##
# regularizaiton by adding w**2
##


def fr(w):
    lambda_ = 0.5
    return(f(w)+lambda_*magnitude(w)**2)
##
# gradient descents, with constant learning rate alpha
##

######################################################
# NUMBER 2
######################################################


def grad_des(g, w, alpha, iter, tol, g0orag1):
    tbeg = process_time()
    count = 1
    updatew = 1.
    errors = []
    while ((count <= iter) and (updatew > tol)):
        if g0orag1 == 0:
            Dg = df(w)
        else:
            gradient = grad(g)
            Dg = gradient(w)
        
        w_old = w
        w = w - alpha*Dg
        updatew = magnitude(g(w) - g(w_old))  # magnitude of update in w
        errors.append(updatew)
        count += 1
    tend = process_time()
    dtime = (tend-tbeg)*1000  # ms
    print('Gradient Descents constant alpha, iter ', count)
    print('delay %.3g ' % dtime, 'ms', 'time/iteration %.3g ' % (dtime/count))
    print('learning rate alpha %.3g ' % alpha)
    print('weight ', w, 'function %.3g' % fabs(w), ' error %.3g' % updatew)
    ploterrors(errors)
    return(w, count, updatew, dtime)

#################################################
# NUMBER 3
#################################################


def grad_desk(g, w, alpha, iter, tol, g0orag1):
    tbeg = process_time()
    count = 1
    
    updatew = 1.
    errors = []
    while ((count <= iter) and (updatew > tol)):
        if g0orag1 == 0:
            Dg = df(w)
        else:
            gradient = grad(g)
            Dg = gradient(w)
        
        w_old = w
        w = w - alpha*Dg / count
        updatew = magnitude(g(w) - g(w_old))  # magnitude of update in w
        errors.append(updatew)
        count += 1
    tend = process_time()
    dtime = (tend-tbeg)*1000  # ms
    print('delay %.3g ' % dtime, 'ms', 'time/iteration %.3g ' % (dtime/count))
    print('Gradient Descents inverse alpha, iter ', count)
    print('learning rate alpha %.3g ' % alpha)
    print('weight ', w, 'function %.3g' % fabs(w), ' error %.3g' % updatew)
    ploterrors(errors)
    return(w, count, updatew, dtime)

#############################################
# NUMBER 4
#############################################


def grad_desn(g, w, alpha, iter, tol, g0orag1):
    tbeg = process_time()
    count = 1
    absDg = 1.0
    eta = 0.01
    updatew = 1.
    errors = []
    while ((count <= iter) and (updatew > tol)):
        if g0orag1 == 0:
            Dg = df(w)
        else:
            gradient = grad(g)
            Dg = gradient(w)
        absDg = magnitude(Dg)
        w_old = w
        w = w - alpha*Dg / (absDg + eta)
        updatew = magnitude(g(w) - g(w_old))  # magnitude of update in w
        errors.append(updatew)
        count += 1
    tend = process_time()
    dtime = (tend-tbeg)*1000  # ms
    print('delay %.3g ' % dtime, 'ms', 'time/iteration %.3g ' % (dtime/count))
    print(f'Gradient Descents Lipshitz {L}, count {count}')
    print('learning rate alpha %.3g ' % alpha)
    print('weight ', w, 'function %.3g' % fabs(w), ' error %.3g' % updatew)
    ploterrors(errors)
    return(w, count, updatew, dtime)

#############################################
# NUMBER 5
#############################################


def lipshitz(w):
    return norm(ddf(w), 2)


def grad_desL(g, w, alpha, iter, tol, g0orag1):
    tbeg = process_time()
    count = 1
    
    updatew = 1.
    errors = []
    while ((count <= iter) and (updatew > tol)):
        if g0orag1 == 0:
            Dg = df(w)
        else:
            gradient = grad(g)
            Dg = gradient(w)
        
        L = lipshitz(w)
        w_old = w
        w = w - alpha*Dg / L
        updatew = magnitude(g(w) - g(w_old))  # magnitude of update in w
        errors.append(updatew)
        count += 1
    tend = process_time()
    dtime = (tend-tbeg)*1000  # ms
    print('delay %.3g ' % dtime, 'ms', 'time/iteration %.3g ' % (dtime/count))
    print('Gradient Descents constant alpha, iter ', count)
    print('learning rate alpha %.3g ' % alpha)
    print('weight ', w, 'function %.3g' % fabs(w), ' error %.3g' % updatew)
    ploterrors(errors)
    return(w, count, updatew, dtime)

##########################################
# NUMBER 6
##########################################

def grad_desm(g, w, alpha, beta, iter, tol, g0orag1):
    tbeg = process_time()
    count = 1
    
    h = np.zeros(w.shape)
    updatew = 1.
    errors = []
    while ((count <= iter) and (updatew > tol)):
        if g0orag1 == 0:
            Dg = f(w)
        else:
            gradient = grad(g)
            Dg = gradient(w)
        
        h = beta * h + (1 - beta) * Dg
        if s == 1:
            alpha = golden(g, w, h, alpha)
            if count % 100 == 0:
                print(f"count {count}\nbeta {beta}\nalpha {alpha:.3g}")
        w_old = w
        w = w - alpha * h
        updatew = magnitude(g(w) - g(w_old))  # magnitude of update in w
        errors.append(updatew)
        count += 1
    tend = process_time()
    dtime = (tend-tbeg)*1000  # ms
    print('delay %.3g ' % dtime, 'ms', 'time/iteration %.3g ' % (dtime/count))
    print('Steepest Descents, momentum beta %.3g' % beta, 'iterations ', count)
    if s == 0: print('learning rate alpha %.3g ' % alpha)
    print('weight ', w, 'function %.3g' % fabs(w), ' error %.3g' % updatew)
    ploterrors(errors)
    return(w, count, updatew, dtime)


###############################################
# NUMBER 7
###############################################

def grad_desq(g, w, alpha, beta, iter, tol, g0orag1):
    tbeg = process_time()
    count = 1
    
    h = np.zeros(w.shape)
    updatew = 1.
    errors = []
    while ((count <= iter) and (updatew > tol)):
        if g0orag1 == 0:
            Dg = f(w)
        else:
            gradient = grad(g)
            Dg = gradient(w)
        
        h = beta * h + (1 - beta) * Dg
        if s == 1:
            alpha = golden(g, w, h, alpha)
            if count % 100 == 0:
                print(f"count {count}\nbeta {beta}\nalpha {alpha:.3g}")
        w_old = w
        w = w - alpha * h
        updatew = magnitude(g(w) - g(w_old))  # magnitude of update in w
        errors.append(updatew)
        count += 1
    tend = process_time()
    dtime = (tend-tbeg)*1000  # ms
    print('delay %.3g ' % dtime, 'ms', 'time/iteration %.3g ' % (dtime/count))
    print('Steepest Descents, momentum beta %.3g' % beta, 'iterations ', count)
    if s == 0: print('learning rate alpha %.3g ' % alpha)
    print('weight ', w, 'function %.3g' % fabs(w), ' error %.3g' % updatew)
    ploterrors(errors)
    return(w, count, updatew, dtime)


#################################################
# NUMBER 8
#################################################

def grad_desj(g, w, alpha, iter, tol, g0orag1):
    tbeg = process_time()
    count = 1
    lambda_ = 0.5
    
    updatew = 1.
    errors = []
    while ((count <= iter) and (updatew > tol)):
        if g0orag1 == 0:
            Dg = f(w)
        else:
            gradient = grad(g)
            Dg = gradient(w)
        
        reg = f(w) + lambda_ * abs(w**2)
        
        w_old = w
        w = w - alpha * Dg / reg
        updatew = magnitude(g(w) - g(w_old))  # magnitude of update in w
        errors.append(updatew)
        count += 1
    tend = process_time()
    dtime = (tend-tbeg)*1000  # ms
    print('delay %.3g ' % dtime, 'ms', 'time/iteration %.3g ' % (dtime/count))
    print('learning rate alpha %.3g ' % alpha)
    print('weight ', w, 'function %.3g' % fabs(w), ' error %.3g' % updatew)
    ploterrors(errors)
    return(w, count, updatew, dtime)




if __name__ == "__main__":
    main()
