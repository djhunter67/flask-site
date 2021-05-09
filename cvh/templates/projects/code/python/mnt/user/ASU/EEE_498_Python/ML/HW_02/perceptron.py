# Christerpher Hunter
# EEE 498 - ML
# HW 02 #2


from colorama import Fore as F
from numpy import dot, array, zeros, ones
from matplotlib.pyplot import plot, title, xlabel, ylabel, grid, savefig

import matplotlib
matplotlib.use('agg')
from sklearn.metrics import accuracy_score


R = F.RESET


def main():
    X = array([
        [-2, 4, -1],
        [4, 1, -1],
        [1, 6, -1],
        [2, 4, -1],
        [6, 2, -1],
    ])

    y = array([-1, -1, 1, 1, 1])

    run_it(X, y)

  



def perceptron_sgd(X, Y, eta):
    w = zeros(len(X[0]))
    time_area = 15
    errors = []
    for time in range(time_area):
        total_error = 0.0
        summed_error = 0.0
        for i, x in enumerate(X):
            error = dot(X[i], w) * Y[i]
            
            if error <= 0:
                dw = X[i] * Y[i]
                w = w + eta * dw
                total_error += error
                print(' Epoch ', time, ' dw ', dw, ' w ', w, ' error ', error)
    
        errors.append(total_error * -1)

    print(' eta ', eta)
    plot(errors)
    title('Error vs Iteration')
    xlabel('iterations')
    ylabel('error')
    grid()
    savefig('error_plot')
    print(f'Final Weights {w}')
    return w


def run_it(X, y):
    w = perceptron_sgd(X, y, 1)
    error_total = 0.0
    for i, x in enumerate(X):
        predicted_y = dot(X[i], w)
        predicted_y_limits = (predicted_y > 0) * 2 - 1
        error = y[i] - predicted_y_limits
        error_total += error
        print(f'\nPredicted: {F.YELLOW}{predicted_y}{R}\n\
Predicted Bin: {F.CYAN}{predicted_y_limits}{R}\n\
Actual: {F.GREEN}{y[i]}{R}\n\
ERROR: {F.RED}{error}{R}')
        print(f'TOTAL ERROR: {F.RED}{error_total}{R}\n')
    print(f'\nFinal Weights: {F.MAGENTA}{w}{R}\n\n')
    return predicted_y_limits

if __name__ == '__main__':
    main()
