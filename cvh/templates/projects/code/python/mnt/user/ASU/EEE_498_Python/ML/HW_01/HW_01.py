# Christerpher Hunter
# HW_01
# EEE 498: Machine Learning



"""
Number 1:
Zeroth: g(w*) <= g(w) for all w

First: the gradient of g(v) = 0 if v is a static point.  If the function is convex this
assures a global minimum.

Second: g''(v) > 0 at convex of the function
g''(v) < 0 at concave portions of the function
saddle points g''(v) = 0 and g''(v) changes sign at those points
"""

"""
Number 2:
Zeroth: g*(lbmda*omega_1 + ((1 - lbmda)*omega_2)) <= lbmda*g(omega_1) + ((1 - lbmda)*g*(omega_2))
0 < lbmda < 1

First: g*(omega) >= g(v) + grad(g(v))(omega - v)
"""

"""
Number 3:
g(W) = (1/P)*sum((X_p^T)*W-y)^2)

show that the second derivative/gradient greater than 0.
"""



## Number 4a
#
from sklearn.metrics import r2_score
from pandas import read_csv
from numpy import array, transpose, random, matmul, linalg, average, sum

# import data
#
df = read_csv('HW_01/regressionprob1_train0.csv')
x_ray = df.iloc[:,0:4].values
yankee = df['F'].values
delta = random.rand(len(yankee))
alpha = transpose(array([x_ray[:,0],x_ray[:,1],x_ray[:,2],x_ray[:,3],delta]))
highest_index = 4

# Least squares training
#
alpha_t_alpha = matmul(transpose(alpha), alpha)
alpha_t_yankee = matmul(transpose(alpha), yankee)
whiskey = matmul(linalg.inv(alpha_t_alpha), alpha_t_yankee)
print(f'Weights from Training: {whiskey}')
yankee_poppa = matmul(alpha, whiskey)

variance_check = variance(yankee, yankee_poppa)
print(f'Training R^2L {r2_score(yankee, yankee_poppa)}.3f')
print(f'Explained Variance: {variance_check}')

# Number 4b
#
whiskey_ = linalg.solve(alpha_t_alpha, alpha_t_yankee)
print(f'Training Weights part_b: {whiskey_}')
def Rsquared(Y,Yp):
    V = Y-Yp
    Ymean = average(Y)
    totvar = sum((Y-Ymean)**2)
    unexpvar = sum(np.abs(V**2))
    R2 = 1-unexpvar/totvar
    return(R2)