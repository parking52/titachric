__author__ = 'melchior'
import numpy as np
from sklearn import linear_model

class Arflou():

    def __init__(self, train_set, train_result):
        self.train_set = train_set
        self.train_result = train_result
        self.clf = linear_model.LinearRegression()

        self.train()

    def train(self):
        self.clf.fit(self.train_set[:, [1, 5]], self.train_result)

    def guess(self, test_set):
        output = self.clf.predict(test_set[:, [1, 5]])
        output[output >= 0.5] = 1
        output[output < 0.5] = 0
        return output




