# Christerpher Hunter
# EEE 498 - ML
# HW_02 Adaline, Quantization, Least Squares Cost Function


from colorama import Fore as F
from numpy import random, ones, column_stack, dot
from pandas import read_csv
from matplotlib.pyplot import plot, title, xlabel, ylabel, grid, savefig
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Perceptron
import matplotlib
matplotlib.use('agg')

R = F.RESET


def main():

    random.seed(1)

    Nfeatures = 3
    data_ = read_csv(
        '/home/djhunter67/winhome/Documents/ASU/EEE_498_Python/ML/HW_02/Dataset_2.csv')
    cols = data_.columns

    X = data_.iloc[:, 0:Nfeatures].values
    y = data_.iloc[:, Nfeatures].values
    lobes = len(X[:, 0])
    standard_scaler = StandardScaler()
    # Normalization Step
    #
    X_std = standard_scaler.fit_transform(X[:, 0:Nfeatures])

    first_ones = ones(lobes, dtype=float)
    X_std_ = column_stack((first_ones, X_std))

    X_train, X_test, y_train, y_test = train_test_split(
        X_std_, y, test_size=0.3, random_state=0)

    w = run_adaline(X_train, y_train, X_test, y_test)
    l_perceptron(X_train, X_test, y_train, y_test, w)


def model(X, w):
    return(dot(X, w))


def CostF(X, w, Y):
    return((model(X, w)-Y)**2)


def dCostF(X, w, Y):
    return((2.*model(X, w)-Y)*X)


def aCostF(X, w, Y):
    error = activation(model(X, w))-Y
    return(error*error)


def activation(Z):
    if (Z >= 0.0):
        return(1)
    else:
        return(-1)


def aldine_sgd(X, w, Y, eta):
    Nobs = len(X[:, 0])
    epochs = 1500
    errors = []
    total_error = 1.0
    tol = 0
    t = 1
    while (t < epochs and total_error > tol):   # stop when the error is 0, or limit in epochs
        total_error = 0.0
        dw = 0.0
        for i in range(int(len(X))):
            # Stochastic Gradient Descents (SGD), choose method
            # create a random number pointing to data
            j = random.randint(Nobs)
            # all the data in the dataset
            error = aCostF(X[j], w, Y[j])      # error from Cost function

            total_error += error            # total error from all cases
            if (error != 0.0):  # this activation step causes it to stop at the solution
                # X[i] is the weight update Y[i] is the direction
                dw += dCostF(X[j], w, Y[j])

        # it only compares to 0 so a constant scale doesn't matter
        w -= eta*dw/(len(X))
        if (t % 100 == 0):
            print(f'epoch {t} total_error {F.RED}{total_error}{R}')
        errors.append(total_error/len(X))  # error from each epoch
        t += 1

    return w, errors, t  # return weight, total error for each epoch, and last epoch


def apply_model(X, w, y):
    ypreds = []
    error_total = 0.0
    for i, x in enumerate(X):
        ypred = activation(model(x, w))

        error = y[i]-ypred
        error_total += error*error
        ypreds.append(ypred)

    # determine accuracy of prediction of the training set
    #
    print(f'Misclassified samples: {F.YELLOW}{(y != ypreds).sum() + 1}{R}')
    print(f'Accuracy: {F.GREEN}{accuracy_score(y, ypreds):2f}{R}')
    return ypreds, error_total


def run_adaline(X_train, y_train, X_test, y_test):
    w = ones(len(X_train[0])) * 0.1
    w, errors, t = aldine_sgd(X_train, w, y_train, 0.001)
    print(f'Weights: {F.MAGENTA}{w}{R}')
    print(f'Epochs: {F.YELLOW}{t}{R}')

    plot(errors)
    title('Error vs Iteration')
    xlabel('iterations')
    ylabel('error')
    grid()
    savefig('adaline')

    ypreds_train, error_total = apply_model(X_train, w, y_train)
    ypreds_test, error_total = apply_model(X_test, w, y_test)
    return w

def l_perceptron(X_train, X_test, y_train, y_test, w):

    # perceptron linear
    print("\nLinear Perceptron SKlearn")

    ppn = Perceptron(max_iter=1500, tol=1e-6, eta0=0.001, random_state=0)
    ppn.fit(X_train, y_train)

    y_pred = ppn.predict(X_train)
    print(f'Training Misclassified samples:  {F.RED}{(y_train != y_pred).sum()}{R}')

    print(f'Train Accuracy: {F.GREEN}{accuracy_score(y_train, y_pred):2f}{R}')

    y_pred = ppn.predict(X_test)
    print(f'Test Misclassified samples: {F.RED}{(y_test != y_pred).sum()}{R}')

    print(f'Test Accuracy: {F.GREEN}{accuracy_score(y_test, y_pred):2f}{R}')

    ##
    # comparison of results
    ##
    print(f'Adaline Weights:  {F.MAGENTA}{w}{R}')
    print(f'perceptron weights: {F.YELLOW}{ppn.intercept_}{R}, {F.YELLOW}{ppn.coef_}{R}')


if __name__ == '__main__':
    main()
