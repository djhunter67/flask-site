## ANITA MALLIK
## 1212667714
## Naive Bayes: using Gaussian fit 
import numpy as np
from sklearn import datasets

iris = datasets.load_iris()
X = iris.data[:,0:4]
y = iris.target

class NaiveBayesGauss():
    
    def _init_(self, X, y):
        self.num_observe, self.num_features = X.shape #row , column
        self.num_class = len(np.unique(y))  #finds length of unique values
        
    
    def fit(self, X, y):
        
        self.classes_mean = {}
        self.classes_stdv = {}
        self.classes_prior = {}
        ## Calculate mean and stdv and classprior= class probabiity for each feature in each class and store in dictionary named by each class
        for c in range (self.num_class):
            X_c = X[y==c]
            self.classes_mean[str(c)] = np.mean(X_c, axis=0)
            self.classes_stdv[str(c)] = np.std(X_c, axis=0)
            self.classes_prior[str(c)] = X_c.shape[0]/self.num_observe #classes_prior will equal the probablilty of the class initially
        
        print('Mean: ', self.classes_mean)
        print('\nStandard Deviation: ',self.classes_stdv)
        print('\nClass Probability: ',self.classes_prior)
        
        
    def predict(self, X):
        prob = np.zeros((self.num_observe, self.num_class))
        #iterate by class and store probabilities of observation for each class
        for c in range (self.num_class):
            prior = self.classes_prior[str(c)]
            mean = self.classes_mean[str(c)]
            stdv = self.classes_stdv[str(c)] 
            prob_c = np.ones((self.num_observe))
            ## iterate each observation and store probablity of each feature in observation for each class
            for n in range (self.num_observe):
                ##store each features Gaussian probability for the observation
                for f in range(self.num_features):
                    prob_c[n] *= self.Gaussian(X[n,f], mean[f], stdv[f])
            ## calculate probabilty of the obsevation for this class by product of all features*classprior probabilty
            prob[:,c] = prob_c * prior
        ##highest probabilty the observation will be in the class will return as its prediction    
        return np.argmax(prob,1)
    
    
    def Gaussian(self, x, mu, sig):
        return np.exp(-np.power(x-mu, 2.) / (2*np.power(sig, 2.) + 1e-300))
        
    
NB= NaiveBayesGauss()
NB._init_(X,y)
NB.fit(X, y)
y_predict = NB.predict(X)

miss = 0
for i in range (len(y)):
    if (y_predict[i] != y[i]):
        print("predicted:", y_predict[i]," actual: ", y[i])
        miss += 1
    else:
        miss = miss 
print("Number of Missclassified: ", miss)

accuracy = 1- (miss/len(y))
print('Accuracy: ', accuracy)
