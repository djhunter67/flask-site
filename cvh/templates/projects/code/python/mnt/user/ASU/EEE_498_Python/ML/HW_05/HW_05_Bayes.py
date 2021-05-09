# Christerpher Hunter
# HW_05: Naive Bayes
# EEE 498

from numpy import log, pi, sum, power, unique, mean, var, zeros, exp
from numpy.core.fromnumeric import argmax, std
from numpy.core.numeric import ones
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score


def main():

    iris = datasets.load_iris()
    X = iris.data[:, 0:4]
    y = iris.target

    NB = Naive_Bayes(X, y)
    NB.__init__(X, y)
    NB.fit(X, y)
    y_pred = NB.predict(X)

    print(f"Number of mislabeled points out of a total {X.shape[0]} \
        \npoints: {sum(y_pred == y)}")

    miss = 0
    for i in range(len(y)):
        if (y_pred[i] != y[i]):
            print(f'predicted: {y_pred[i]} actual:  {y[i]}')
            miss += 1
        else:
            miss = miss
    print(f'Number of Missclassified: {miss}')

    accuracy = 1 - (miss/len(y))
    print(f'Accuracy: {accuracy} -- SELF IMPLEMENTED\n')

############################################################
# sklearn Naive Bayes
############################################################
    gnb = GaussianNB()

    # Trained the entire database
    #
    y_predicted = gnb.fit(X, y).predict(X)
    print(f"Number of mislabeled points out of a total {X.shape[0]} \
        \npoints: {(y != y_predicted).sum()} -- SKLEARN IMPLEMENTED")
    print(f'Accuracy: {accuracy_score(y, y_predicted)}')



class Naive_Bayes():
    def __init__(self, X, y):
        self.num_observations, self.num_features = X.shape
        self.num_classes = len(unique(y))
       
    def fit(self, X, y):
        self.classes_mean = {}
        self.classes_stdv = {}
        self.classes_prior = {}

        for c in range(self.num_classes):
            X_c = X[y == c]

            self.classes_mean[str(c)] = mean(X_c, axis=0)
            self.classes_stdv[str(c)] = std(X_c, axis=0)
            self.classes_prior[str(c)] = X_c.shape[0] / self.num_observations

    def predict(self, X):
        probs = zeros((self.num_observations, self.num_classes))

        for c in range(self.num_classes):
            prior = self.classes_prior[str(c)]
            mean = self.classes_mean[str(c)]
            stdv = self.classes_stdv[str(c)]
            probs_c = ones((self.num_observations))
            
            for k in range(self.num_observations):
                for q in range(self.num_features):
                    probs_c[k] *= self.PGauss(X[k,q], mean[q], stdv[q])
                probs[:, c] = probs_c * prior

            return argmax(probs, 1)    

    def PGauss(self, mu, sig, x) -> float:
        return exp(-power(x - mu, 2.) / (2 * power(sig, 2.) + 1e-300))


if __name__ == '__main__':
    main()
