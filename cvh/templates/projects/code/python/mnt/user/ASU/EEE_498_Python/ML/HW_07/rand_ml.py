import warnings
import lightgbm as lgb
import time
from matplotlib.colors import Normalize
import torch
import numpy as np
from torch.autograd import Variable
import torch.nn as nn
from scipy.special import expit, logit
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from colorama import Fore as F
from sklearn.decomposition import PCA
from sklearn.svm import SVC

R = F.RESET


def main():


    n_samples = 2000
    random_state = np.random.RandomState(13)
    x1 = random_state.uniform(size=n_samples)
    x2 = random_state.uniform(size=n_samples)
    x3 = random_state.randint(0, 4, size=n_samples)
    x4 = random_state.uniform(size=n_samples)  # note that this is noise
    ##
    ## random.RandomState.binomial(n, p, size=None)
    # n number of trials
    # N number of successes    
    # P(N) = (n N)p^N (1-p)^(n-N)
    ##
    p = expit(np.sin(3 * x1) - 4 * x2 + x3)
    y = random_state.binomial(1, p, size=n_samples)

    X_list = []
    for i in range(0, len(x1)):
        X_list.append([x1[i],x2[i],x3[i],x4[i]])

    X = np.array(X_list)
        
    X_train, y_train, X_test, y_test = prepare_data(X, y)

    # print(f"Training: {X_train.size, y_train.size}\nTesting: {X_test.size, y_test.size}")

    ####################################################
    # Number 1 - Logistic Regression                   #
    ####################################################

    tbeg = time.time()
    logistic_regr(X_train, y_train, X_test, y_test)
    tend = time.time()
    print(f"total training time: {F.LIGHTYELLOW_EX}{round(tend-tbeg, 4)}{R} seconds")
    
    ####################################################
    # Number 2 - SVM Linear Kernel                     #            
    ####################################################

    tbeg = time.time()
    support_vector_machine_kern(X_train, y_train, X_test, y_test, 10)
    tend = time.time()
    print(f"total training time: {F.LIGHTYELLOW_EX}{round(tend-tbeg, 4)}{R} seconds")

    ####################################################
    # Number 3 - Light Gradient Boosting Method        #
    ####################################################
    
    display_header("LEAST GRADIENT BOOSTING METHOD")
    warnings.filterwarnings("ignore")
    tbeg = time.time()
    lgb_train = lgb.Dataset(X_train, y_train)
    lgb_test = lgb.Dataset(X_test, y_test)
    # defaults num_leaves = 31,
    params = {'force_col_wise': True, 'boosting_type': 'gbdt', 'num_iterations': 100,
            'n_estimators': 100,
            'max_depth': 10, 'num_leaves': 100, 'feature_fraction': 0.75,
            'bagging_fraction': 0.75, 'bagging_freq': 1, 'lambda': 0.10, 'random_state': 3}
    model = lgb.train(params, lgb_train, valid_sets=[
                    lgb_train, lgb_test], verbose_eval=20)
    print('Number in train ', len(y_train))

    print('Number in train ', len(y_train))
    y_train_pred = model.predict(X_train)
    y_pred = np.where(y_train_pred < 0.10, 0, 1)
    # acc_test = accuracy_score(y_test, y_pred, normalize=False)
    # print(f'Accuracy: {y_pred}')
    ones = 0
    zero = 0
    for k in y_pred:
        if y_pred[k] == 1:
            ones += 1
        else:
            zero += 1

    print(f"ONES: {ones}\nZEROES: {zero}")
        
    tend = time.time()
    print(f"total training time: {F.LIGHTYELLOW_EX}{round(tend-tbeg, 4)}{R} seconds")
    
    ###################################################
    # Number 4 - Two Layer Neural Network (sequential)#
    ###################################################
    """
    XX_train = torch.from_numpy(X_train)
    # targets = y_train.astype(int)    ## cast to int
    targets0 = np.eye(7)[y_train.astype(int)]   ## one hot code target
    yy_train = torch.from_numpy(np.eye(7)[y_train.astype(int)]).type(torch.FloatTensor)
    """


def display_header(header):
    """Succint description of the program"""
    print(f'\n{F.GREEN}------------------------------------------')
    print(f'                {header}')
    print(f'------------------------------------------{R}\n')

def prepare_data(datax, datay):
            
        # split the problem into train and test
        X_train, X_test, y_train, y_test = train_test_split(
                                                            datax, 
                                                            datay, 
                                                            test_size=0.30,
                                                            random_state=0)

        # scale X
        sc = StandardScaler()
        sc.fit(X_train)
        X_train_std = sc.transform(X_train)
        X_test_std = sc.transform(X_test)

        return X_train_std, y_train, X_test_std, y_test

def p_c_a(X_train, X_test_std, n_components=0):
    """PRICIPLE COMPONENT ANALYSIS"""

    pca = PCA(n_components, svd_solver='full')
    pca.fit(X_train)

    X_train_std = pca.transform(X_train)
    X_test_std = pca.transform(X_test_std)

    X_train_pca = pca.fit_transform(X_train_std)
    X_test_pca = pca.transform(X_test_std)

    # print('\nPRICIPLE COMPONENT ANALYSIS')

    # print("Variance Explained", pca.explained_variance_ratio_)
    # print("Total Variance Explained", sum(pca.explained_variance_ratio_))
    # print(f'The total number of "variance explained" values:\
    # {len(pca.explained_variance_ratio_)}')

    return X_train_pca, X_test_pca

def logistic_regr(X_train_std, y_train, X_test_std, y_test, n_comp=0):
    """EXECUTE LOGISTIC REGRESSION"""

    # The larger C, the larger the penalty for fitting error,
    # very sensitive to C
    lr = LogisticRegression(max_iter=10, tol=1e-4, C=10,
                            solver='liblinear', random_state=0)

    

    lr.fit(X_train_std, y_train)

    print(f'\n{F.CYAN}LOGISTIC REGRESSION{R}')
    print('Number in train ', len(y_train))
    y_pred = lr.predict(X_train_std)
    print('Misclassified train samples: %d' % (y_train != y_pred).sum())

    train_acc = accuracy_score(y_train, y_pred)
    print(f'{F.GREEN}Train Accuracy: {R}%.2f' % train_acc)

    print('Number in test ', len(y_test))
    y_pred = lr.predict(X_test_std)
    print('Misclassified test samples: %d' % (y_test != y_pred).sum())

    test_acc = accuracy_score(y_test, y_pred)
    print(f'{F.GREEN}Test Accuracy:{R} {F.RED}%.2f{R}' % test_acc)

    X_combined_pca = np.vstack((X_train_std, X_test_std))
    y_combined = np.hstack((y_train, y_test))
    print('Number in combined ', len(y_combined))

    y_combined_pred = lr.predict(X_combined_pca)
    print('Misclassified combined samples: %d' %
          (y_combined != y_combined_pred).sum())

    percentage = ((abs(round(train_acc, 7) - round(test_acc, 7))
                   / (round(train_acc, 7) + round(test_acc, 7)) / 2) * 100)

    print(f'{F.RED}Overfit: {percentage:.2f}{R}')

    return 'LOGISTIC_REGRESSION', accuracy_score(y_test, y_pred), n_comp,\
        percentage

def support_vector_machine_kern(X_train_std,
                                y_train,
                                X_test_std,
                                y_test,
                                n_C=1,
                                n_comp=0):
    """EXECUTE SUPPORT VECTOR MACHINE - LINEAR"""
    
    # Support Vector Machine
    svm = SVC(kernel='linear', C=n_C / 10, random_state=0, gamma=10.0)
    svm.fit(X_train_std, y_train)

    print(f'\n{F.CYAN}SUPPORT VECTOR MACHINE - LINEAR{R}')
    print('Number in train ', len(y_train))
    y_pred = svm.predict(X_train_std)
    print('Misclassified train samples: %d' % (y_train != y_pred).sum())

    train_acc = accuracy_score(y_train, y_pred)
    print(f'{F.GREEN}Train Accuracy: {R}%.2f' % train_acc)

    print('Number in test ', len(y_test))
    y_pred = svm.predict(X_test_std)
    print('Misclassified test samples: %d' % (y_test != y_pred).sum())

    test_acc = accuracy_score(y_test, y_pred)
    print(f'{F.GREEN}Test Accuracy:{R} {F.RED}%.2f{R}' % test_acc)

    X_combined_pca = np.vstack((X_train_std, X_test_std))
    y_combined = np.hstack((y_train, y_test))
    print('Number in combined ', len(y_combined))

    y_combined_pred = svm.predict(X_combined_pca)
    print('Misclassified combined samples: %d' %
          (y_combined != y_combined_pred).sum())

    print('Combined Accuracy: %.2f' %
          accuracy_score(y_combined, y_combined_pred))

    percentage = ((abs(round(train_acc, 10) - round(test_acc, 10))
                   / (round(train_acc, 10) + round(test_acc, 10)) / 2) * 100)

    print(f'{F.RED}Overfit: {percentage:.2f}{R}')

    return 'KVM_KERNEL', accuracy_score(y_test, y_pred), n_comp,\
        percentage

if __name__ == "__main__":
    main()