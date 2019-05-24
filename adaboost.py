# coding=utf-8
import numpy as np
from weakClassifier import WeakClassifier

from setting import FACE, NON_FACE

class Adaboost(object):
    def __init__(self, n_estimators = 100, debug=False):
        self.n_estimators = n_estimators
        self.weakClassifiers = [None for i in range(self.n_estimators)]
        self.alpha = np.zeros(n_estimators)
        self.debug = debug

    def fit(self, X, Y):
        """
        :param X:
        :param Y: shape:sampleNum * 1
        """
        sampleNum = X.shape[0]
        # weight shape:
        weight = np.array([1/sampleNum for i in range(sampleNum)]).reshape(-1, 1)
        for i in range(self.n_estimators):
            if self.debug:
                print("training " + str(i) + "th weakClassifier....")
            self.weakClassifiers[i] = WeakClassifier()
            weightError = self.weakClassifiers[i].fit(X, weight, Y)
            self.alpha[i] = (1/2) * np.log((1 - weightError) / weightError)
            output = self.weakClassifiers[i].predict(X)
            from sklearn.metrics import accuracy_score
            print(accuracy_score(Y, output))
            Z = weight * np.exp(-self.alpha[i] * Y * output)
            weight = Z / Z.sum()

    def predict(self, X):

        pred = np.zeros((X.shape[0], 1))
        for i in range(self.n_estimators):
            weakOutput = self.weakClassifiers[i].predict(X)
            pred = pred + weakOutput * self.alpha[i]
        pred[np.where(pred > 0) ] = FACE
        pred[np.where(pred <= 0)] = NON_FACE

        return pred

    def predict_prob(self, X):
        """return the probability of each sample
        """
        pred = np.zeros((X.shape[0], 1), dtype='float32')
        for i in range(self.n_estimators):
            weakOutput = self.weakClassifiers[i].predict(X)
            pred = pred + weakOutput * self.alpha[i]

        return pred




