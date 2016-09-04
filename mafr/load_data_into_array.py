__author__ = 'melchior'


import csv
import os
import numpy as np

train_data = np.genfromtxt(os.path.join('..', 'data', 'train.csv'), delimiter=',')

columns = ['PassengerId', 'Pclass', 'Name', 'Sex', 'Age', 'SibSp', 'Parch', 'Ticket', 'Fare', 'Cabin', 'Embarked']
df = [train_data, columns]

print(df[0])

test_set = np.genfromtxt(os.path.join('..', 'data', 'test.csv'), delimiter=',')
test_columns = ['PassengerId', 'Pclass', 'Name', 'Sex', 'Age', 'SibSp', 'Parch', 'Ticket', 'Fare', 'Cabin','Embarked']

train_set = np.delete(train_data, 1, axis=1)
train_result = train_data[:, 1]
test_set = test_set
test_guess = np.zeros(train_set.shape[0])
pass