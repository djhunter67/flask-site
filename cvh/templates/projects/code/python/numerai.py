#!/usr/bin/env python3

"""Use Machine Learning to detect a rock or an underwater mine"""

from colorama import Fore as F
from pandas import read_csv
from numpy import tri, abs, arange, array, vstack, hstack, float16
from matplotlib import cm as cm
from matplotlib.pyplot import title, figure, show
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Perceptron
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition import PCA
import csv
import torch

R = F.RESET
TEST_PERC = 0.33
RAND_STATE = 0

if torch.cuda.is_available():  
    dev = "cuda:0" 
else:  
    dev = "cpu"  
    device = torch.device(dev)


def main():
    display_header('NUMERAI')

    data = read_file('numerai_training_data.csv')
     
    # print(f"data:\n {data}")    
    
    # analyze_data(data, len(data))

    # correl_matrix(data, 10)   
    
    X_train_std, y_train, X_test_std, y_test = prepare_data(data)
    y_train = y_train.astype('int16')
    y_test = y_test.astype('int16')
    
    
    test_accuracy = []
    for i in range(1, 70):
        EIGENS = i
        print(f'\nPCA: {F.YELLOW}{EIGENS}{R}')
        
        log_reg = logistic_regr(X_train_std,
                                y_train,
                                X_test_std,
                                y_test,
                                n_comp=EIGENS)
        lin_perc = linear_perceptron(X_train_std,
                                     y_train,
                                     X_test_std,
                                     y_test,
                                     n_comp=EIGENS)
        svm_lin = support_vector_machine_lin(X_train_std,
                                             y_train,
                                             X_test_std,
                                             y_test,
                                             n_comp=EIGENS)
        dec_tree = decision_tree(X_train_std,
                                 y_train,
                                 X_test_std,
                                 y_test,
                                 n_comp=EIGENS)
        r_forest = random_forest(X_train_std,
                                 y_train,
                                 X_test_std,
                                 y_test,
                                 n_est=10,
                                 n_comp=EIGENS)
                                
        knn = k_nearest(X_train_std, y_train, X_test_std,
                        y_test, n_comp=EIGENS)
        
        svm_kern = support_vector_machine_kern(X_train_std,
                                    y_train,
                                    X_test_std,
                                    y_test,
                                    n_C=EIGENS,
                                    n_comp=310)
        test_accuracy.append(log_reg[0:3])
        test_accuracy.append(lin_perc[0:3])
        test_accuracy.append(svm_lin[0:3])
        test_accuracy.append(dec_tree[0:3])
        test_accuracy.append(r_forest[0:3])
        #test_accuracy.append(knn[0:3])
        #test_accuracy.append(svm_kern[0:3])

    print(f'\n{F.YELLOW}{max(test_accuracy)}{R}\n')
    


def display_header(header):
    """Succint description of the program"""
    print(f'\n{F.GREEN}------------------------------------------')
    print(f'                {header}')
    print(f'------------------------------------------{R}\n')
    

def df_to_tensor(df):
    
    return torch.from_numpy(df.values).float()


def read_file(filename):
    cols_list = [x + 3 for x in range(311)]

    with open(filename) as fin:
        column_names = next(csv.reader(fin))

    dtypes = {x: float16 for x in column_names if x.startswith(('feature', 'target'))}
    df = read_csv(filename, dtype=dtypes, usecols=cols_list, header=0)# , nrows=500)
    
    return df


def correl_matrix(X, cols):
    """ Covariance/Correlation matrix """

    fig = figure(figsize=(7, 7), dpi=100)
    ax1 = fig.add_subplot(111)
    cmap = cm.get_cmap('jet', 30)
    cax = ax1.imshow(abs(X.corr()), interpolation='nearest', cmap=cmap)
    # ax1.set_xticks(major_ticks)
    major_ticks = arange(0, len(cols), 1)
    ax1.set_xticks(major_ticks)
    ax1.set_yticks(major_ticks)
    ax1.grid(True, which='both', axis='both')
    # plt.aspect('equal')
    title('Correlation Matrix')
    labels = cols
    ax1.set_xticklabels(labels, fontsize=9)
    ax1.set_yticklabels(labels, fontsize=12)
    fig.colorbar(cax, ticks=[-0.4, -0.25, -.1, 0, 0.1, .25, .5, .75, 1])

    return(1)


def analyze_data(data, numtoreport):
    cols = data.columns

    # descriptive statistics
    print(' Descriptive Statistics ')
    print(data.describe())

    # heat plot of covariance
    print(' Covariance Matrix \n')
    correl_matrix(data.iloc[:, 0:numtoreport], cols[0:numtoreport])

    """create covariance for dataframes"""

    # find the correlations
    cormatrix = data.corr()
    # set the correlations on the diagonal or lower triangle to zero,
    # so they will not be reported as the highest ones:
    cormatrix *= tri(*cormatrix.values.shape, k=-1).T
    # find the top n correlations
    cormatrix = cormatrix.stack()
    cormatrix = cormatrix.reindex(cormatrix.abs().sort_values(
        ascending=False).index).reset_index()
    # assign human-friendly names
    cormatrix.columns = ["FirstVariable", "SecondVariable", "Correlation"]

    # show()
    return cormatrix.head(numtoreport)


def prepare_data(data):
    x_list = [x for x in range(data.shape[1])]
    X = array(data.iloc[:, : - 1])
    y = array(data.iloc[:, -1])

    # split the problem into train and test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_PERC, random_state=RAND_STATE)

    # scale X
    sc = StandardScaler()
    sc.fit(X_train)
    X_train_std = sc.transform(X_train)
    X_test_std = sc.transform(X_test)

    return X_train_std, y_train, X_test_std, y_test


"""MACHINE LEARNING ALGORITHMS"""


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
                            solver='liblinear', random_state=RAND_STATE)

    X_train_pca, X_test_pca = p_c_a(X_train_std,
                                    X_test_std,
                                    n_components=n_comp)

    lr.fit(X_train_pca, y_train)

    print(f'\n{F.CYAN}LOGISTIC REGRESSION{R}')
    print('Number in train ', len(y_train))
    y_pred = lr.predict(X_train_pca)
    print('Misclassified samples: %d' % (y_train != y_pred).sum())

    train_acc = accuracy_score(y_train, y_pred)
    print(f'{F.GREEN}Train Accuracy:{R}%.2f' % train_acc)

    print('Number in test ', len(y_test))
    y_pred = lr.predict(X_test_pca)
    print('Misclassified samples: %d' % (y_test != y_pred).sum())

    test_acc = accuracy_score(y_test, y_pred)
    print(f'{F.GREEN}Test Accuracy:{R} {F.RED}%.2f{R}' % test_acc)

    X_combined_pca = vstack((X_train_pca, X_test_pca))
    y_combined = hstack((y_train, y_test))
    print('Number in combined ', len(y_combined))

    y_combined_pred = lr.predict(X_combined_pca)
    print('Misclassified combined samples: %d' %
          (y_combined != y_combined_pred).sum())

    percentage = ((abs(round(train_acc, 2) - round(test_acc, 2))
                   / (round(train_acc, 2) + round(test_acc, 2)) / 2) * 100)

    print(f'{F.RED}Overfit: {percentage:.2f}{R}')

    return 'LOGISTIC_REGRESSION', accuracy_score(y_test, y_pred), n_comp,\
        percentage


def linear_perceptron(X_train_std,
                      y_train,
                      X_test_std,
                      y_test,
                      n_comp=0):
    """PERCEPTRON LINEAR CLASSIFIER"""

    X_train_pca, X_test_pca = p_c_a(X_train_std,
                                    X_test_std,
                                    n_components=n_comp)

    ppn = Perceptron(max_iter=40, tol=1e-3, eta0=1, random_state=RAND_STATE)
    ppn.fit(X_train_pca, y_train)

    print(f'\n{F.CYAN}PERCEPTRON - LINEAR{R}')
    print('Number in train ', len(y_train))
    y_pred = ppn.predict(X_train_pca)
    print('Misclassified samples: %d' % (y_train != y_pred).sum())

    train_acc = accuracy_score(y_train, y_pred)
    print(f'{F.GREEN}Train Accuracy:{R}%.2f' % train_acc)

    print('Number in test ', len(y_test))
    y_pred = ppn.predict(X_test_pca)
    print('Misclassified samples: %d' % (y_test != y_pred).sum())

    test_acc = accuracy_score(y_test, y_pred)
    print(f'{F.GREEN}Test Accuracy:{R} {F.RED}%.2f{R}' % test_acc)

    X_combined_pca = vstack((X_train_pca, X_test_pca))
    y_combined = hstack((y_train, y_test))
    print('Number in combined ', len(y_combined))

    y_combined_pred = ppn.predict(X_combined_pca)
    print('Misclassified combined samples: %d' %
          (y_combined != y_combined_pred).sum())

    percentage = ((abs(round(train_acc, 2) - round(test_acc, 2))
                   / (round(train_acc, 2) + round(test_acc, 2)) / 2) * 100)

    print(f'{F.RED}Overfit: {percentage:.2f}{R}')

    # X_combined_std = vstack((X_train_std, X_test_std))
    # y_combined = hstack((y_train, y_test))

    # plot_decision_regions(X=X_combined_std, y=y_combined,
    # classifier=ppn, test_idx=range(105, 150))
    # plt.xlabel('petal length [standardized]')
    # plt.ylabel('petal width [standardized]')
    # plt.legend(loc='upper left')
    # plt.show()

    return 'PERCEPTRON_LINEAR', accuracy_score(y_test, y_pred), n_comp,\
        percentage


def support_vector_machine_kern(X_train_std,
                                y_train,
                                X_test_std,
                                y_test,
                                n_C=1,
                                n_comp=0):
    """EXECUTE SUPPORT VECTOR MACHINE - KERNEL"""

    X_train_pca, X_test_pca = p_c_a(X_train_std,
                                    X_test_std,
                                    n_components=n_comp)

    # Support Vector Machine
    svm = SVC(kernel='rbf', C=n_C / 10, random_state=RAND_STATE, gamma=10.0)
    svm.fit(X_train_std, y_train)

    print(f'\n{F.CYAN}SUPPORT VECTOR MACHINE - KERNEL{R}')
    print('Number in train ', len(y_train))
    y_pred = svm.predict(X_train_pca)
    print('Misclassified samples: %d' % (y_train != y_pred).sum())

    train_acc = accuracy_score(y_train, y_pred)
    print(f'{F.GREEN}Train Accuracy:{R}%.2f' % train_acc)

    print('Number in test ', len(y_test))
    y_pred = svm.predict(X_test_pca)
    print('Misclassified samples: %d' % (y_test != y_pred).sum())

    test_acc = accuracy_score(y_test, y_pred)
    print(f'{F.GREEN}Test Accuracy:{R} {F.RED}%.2f{R}' % test_acc)

    X_combined_pca = vstack((X_train_pca, X_test_pca))
    y_combined = hstack((y_train, y_test))
    print('Number in combined ', len(y_combined))

    y_combined_pred = svm.predict(X_combined_pca)
    print('Misclassified combined samples: %d' %
          (y_combined != y_combined_pred).sum())

    print('Combined Accuracy: %.2f' %
          accuracy_score(y_combined, y_combined_pred))

    percentage = ((abs(round(train_acc, 2) - round(test_acc, 2))
                   / (round(train_acc, 2) + round(test_acc, 2)) / 2) * 100)

    print(f'{F.RED}Overfit: {percentage:.2f}{R}')

    return 'KVM_KERNEL', accuracy_score(y_test, y_pred), n_comp,\
        percentage


def support_vector_machine_lin(X_train_std,
                               y_train,
                               X_test_std,
                               y_test,
                               n_comp=0):
    """EXECUTE SUPPORT VECTOR MACHINE - LINEAR"""

    X_train_pca, X_test_pca = p_c_a(X_train_std,
                                    X_test_std,
                                    n_components=n_comp)

    # Support Vector Machine
    svm = SVC(kernel='linear', C=1.0, random_state=RAND_STATE, gamma=0.1)
    svm.fit(X_train_pca, y_train)

    print(f'\n{F.CYAN}SUPPORT VECTOR MACHINE - LINEAR{R}')
    print('Number in train ', len(y_train))
    y_pred = svm.predict(X_train_pca)
    print('Misclassified samples: %d' % (y_train != y_pred).sum())

    train_acc = accuracy_score(y_train, y_pred)
    print(f'{F.GREEN}Train Accuracy:{R}%.2f' % train_acc)

    print('Number in test ', len(y_test))
    y_pred = svm.predict(X_test_pca)
    print('Misclassified samples: %d' % (y_test != y_pred).sum())

    test_acc = accuracy_score(y_test, y_pred)
    print(f'{F.GREEN}Test Accuracy:{R} {F.RED}%.2f{R}' % test_acc)

    X_combined_pca = vstack((X_train_pca, X_test_pca))
    y_combined = hstack((y_train, y_test))
    print('Number in combined ', len(y_combined))

    y_combined_pred = svm.predict(X_combined_pca)
    print('Misclassified combined samples: %d' %
          (y_combined != y_combined_pred).sum())

    percentage = ((abs(round(train_acc, 2) - round(test_acc, 2))
                   / (round(train_acc, 2) + round(test_acc, 2)) / 2) * 100)

    print(f'{F.RED}Overfit: {percentage:.2f}{R}')
    return 'KVM_LINEAR', accuracy_score(y_test, y_pred), n_comp,\
        percentage


def decision_tree(X_train_std, y_train, X_test_std, y_test, n_comp=0):
    """DECISION TREE EXECUTION"""

    X_train_pca, X_test_pca = p_c_a(X_train_std,
                                    X_test_std,
                                    n_components=n_comp)

    tree_model = DecisionTreeClassifier(criterion='gini', max_depth=8,
                                        random_state=RAND_STATE)
    tree_model.fit(X_train_pca, y_train)

    print(f'\n{F.CYAN}DECISION TREE - DEPTH: {tree_model.get_depth()}{R}')
    print('Number in train ', len(y_train))
    y_pred = tree_model.predict(X_train_pca)
    print('Misclassified samples: %d' % (y_train != y_pred).sum())

    train_acc = accuracy_score(y_train, y_pred)
    print(f'{F.GREEN}Train Accuracy:{R}%.2f' % train_acc)

    print('Number in test ', len(y_test))
    y_pred = tree_model.predict(X_test_pca)
    print('Misclassified samples: %d' % (y_test != y_pred).sum())

    test_acc = accuracy_score(y_test, y_pred)
    print(f'{F.GREEN}Test Accuracy:{R} {F.RED}%.2f{R}' % test_acc)

    X_combined_pca = vstack((X_train_pca, X_test_pca))
    y_combined = hstack((y_train, y_test))
    print('Number in combined ', len(y_combined))

    y_combined_pred = tree_model.predict(X_combined_pca)
    print('Misclassified combined samples: %d' %
          (y_combined != y_combined_pred).sum())

    percentage = ((abs(round(train_acc, 2) - round(test_acc, 2))
                   / (round(train_acc, 2) + round(test_acc, 2)) / 2) * 100)

    print(f'{F.RED}Overfit: {percentage:.2f}{R}')

    # from sklearn import tree
    # tree.plot_tree(tree_model)
    # plt.show()

    return 'DECISION_TREE', accuracy_score(y_test, y_pred), n_comp,\
        percentage


def random_forest(X_train_std, y_train, X_test_std, y_test, n_est=1, n_comp=0):
    """RANDOM FOREST EXECUTION"""

    X_train_pca, X_test_pca = p_c_a(X_train_std,
                                    X_test_std,
                                    n_components=n_comp)

    forest = RandomForestClassifier(criterion='gini',
                                    max_depth=None,
                                    n_estimators=n_est * 2,
                                    random_state=RAND_STATE,
                                    n_jobs=12,
                                    bootstrap=True,
                                    ccp_alpha=0.0468)

    forest.fit(X_train_pca, y_train)

    print(f'\n{F.CYAN}RANDOM FOREST{R}')
    print('Number in train ', len(y_train))
    y_pred = forest.predict(X_train_pca)
    print('Misclassified samples: %d' % (y_train != y_pred).sum())

    train_acc = accuracy_score(y_train, y_pred)
    print(f'{F.GREEN}Train Accuracy:{R}%.2f' % train_acc)

    print('Number in test ', len(y_test))
    y_pred = forest.predict(X_test_pca)
    print('Misclassified samples: %d' % (y_test != y_pred).sum())

    test_acc = accuracy_score(y_test, y_pred)
    print(f'{F.GREEN}Test Accuracy:{R} {F.RED}%.2f{R}' % test_acc)

    X_combined_pca = vstack((X_train_pca, X_test_pca))
    y_combined = hstack((y_train, y_test))
    print('Number in combined ', len(y_combined))

    y_combined_pred = forest.predict(X_combined_pca)
    print('Misclassified combined samples: %d' %
          (y_combined != y_combined_pred).sum())

    percentage = ((abs(round(train_acc, 2) - round(test_acc, 2))
                   / (round(train_acc, 2) + round(test_acc, 2)) / 2) * 100)

    print(f'{F.RED}Overfit: {percentage:.2f}{R}')

    return 'RANDOM_FOREST', accuracy_score(y_test, y_pred), n_comp,\
        n_est


def k_nearest(X_train_std, y_train, X_test_std, y_test, n_comp=0):
    """K NEAREST NEIGHBOR EXECUTION"""

    X_train_pca, X_test_pca = p_c_a(X_train_std,
                                    X_test_std,
                                    n_components=n_comp)

    knn = KNeighborsClassifier(n_neighbors=34, p=1,
                               metric='minkowski')

    knn.fit(X_train_pca, y_train)

    print(f'\n{F.CYAN}K NEAREST NEIGHBOR{R}')
    print('Number in train ', len(y_train))
    y_pred = knn.predict(X_train_pca)
    print('Misclassified samples: %d' % (y_train != y_pred).sum())

    train_acc = accuracy_score(y_train, y_pred)
    print(f'{F.GREEN}Train Accuracy:{R}%.2f' % train_acc)

    print('Number in test ', len(y_test))
    y_pred = knn.predict(X_test_pca)
    print('Misclassified samples: %d' % (y_test != y_pred).sum())

    test_acc = accuracy_score(y_test, y_pred)
    print(f'{F.GREEN}Test Accuracy:{R} {F.RED}%.2f{R}' % test_acc)

    X_combined_pca = vstack((X_train_pca, X_test_pca))
    y_combined = hstack((y_train, y_test))
    print('Number in combined ', len(y_combined))

    y_combined_pred = knn.predict(X_combined_pca)
    print('Misclassified combined samples: %d' %
          (y_combined != y_combined_pred).sum())

    percentage = ((abs(round(train_acc, 2) - round(test_acc, 2))
                   / (round(train_acc, 2) + round(test_acc, 2)) / 2) * 100)

    print(f'{F.RED}Overfit: {percentage:.2f}{R}')

    return 'K_NEAREST_NEIGHBOR', accuracy_score(y_test, y_pred), n_comp,\
        percentage


if __name__ == '__main__':
    main()
