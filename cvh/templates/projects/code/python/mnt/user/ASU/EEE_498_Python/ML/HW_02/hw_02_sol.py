import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('agg')

# observations for 3 variables x0, x1, x2
X = np.array([
    [-2, 4, -1],
    [4, 1, -1],
    [1, 6, -1],
    [2, 4, -1],
    [6, 2, -1],
])
##
# this is a single classificaiton
##
## X = np.concatenate((X,X))
# target
y = np.array([-1, -1, 1, 1, 1])
## y = np.concatenate((y,y))
# our perceptron - stochastic gradient descent


def perceptron_sgd(X, Y, eta):
    w = np.zeros(len(X[0]))
    epochs = 15
    errors = []
    for t in range(epochs):
        total_error = 0.0
# enumerate is used to number the parameters in a list
        for i, x in enumerate(X):
            ##            print(' epoch ', t,' X ', X[i],' Y ', Y[i])
            # will be neg if there is a missclassification
            # x dot w is the pred of y, times y give pos or neg sign
            # to drive the weight higher or lower
            error = np.dot(X[i], w)*Y[i]
            if (error <= 0):  # neg means a miss, one is positive, and one is negative
                # X[i] is the weight update Y[i] is the direction
                dw = X[i]*Y[i]
                w = w + eta*dw  # it only compares to 0 so a constant scale doesn't matter
                total_error += error
                print(' Epoch ', t, ' dw ', dw, ' w ', w, ' error ', error)
##        print(' epoch ', t,' w ', w,'\n')
# for i, x in enumberate(X)
        errors.append(total_error*-1)

    print(' eta ', eta)
    plt.plot(errors)
    plt.title('Error vs Iteration')
    plt.xlabel('iterations')
    plt.ylabel('error')
    plt.grid()
    plt.savefig('solution_plot')
    print(' final weights ', w)
    return w


# run perceptron
w = perceptron_sgd(X, y, 1)
# Binarizer creates a transform that binary at threshold, in this case 0.0
error_total = 0.0
for i, x in enumerate(X):
    ypred = np.dot(X[i], w)
# we want 1 and -1 not 1 and 0 so adjust limits
    ypredp = (ypred > 0)*2-1
    error = y[i]-ypredp
    error_total += error
    print(' predicted ', ypred, ' pred bin ',
          ypredp, ' actual ', y[i], ' error ', error)
print(' Total Error ', error_total)
print(w)
