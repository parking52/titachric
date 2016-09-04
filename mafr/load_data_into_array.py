__author__ = 'melchior'


import csv
import os
import numpy as np

from mafr.arflou import Arflou
from mafr.valid import Valid

data = np.genfromtxt(os.path.join('..', 'data', 'train.csv'), delimiter=',')

columns = ['PassengerId', 'Pclass', 'Name', 'Sex', 'Age', 'SibSp', 'Parch', 'Ticket', 'Fare', 'Cabin', 'Embarked']
df = [data, columns]

print(df[0])

test_set = np.genfromtxt(os.path.join('..', 'data', 'test.csv'), delimiter=',')
test_columns = ['PassengerId', 'Pclass', 'Name', 'Sex', 'Age', 'SibSp', 'Parch', 'Ticket', 'Fare', 'Cabin', 'Embarked']


### Process initial data
## Process Age => default value
default_value = 25
nans_ids = np.where(np.isnan(data[:, 6]))
data[nans_ids, 6] = default_value

train_result = data[1:600, 1]
test_truth = data[600:, 1]

train_set = np.delete(data, 1, axis=1)[1:600]
test_set = np.delete(data, 1, axis=1)[600:]

test_guess = np.zeros(test_set.shape[0])  ##model1 = everyone is dead
test_guess2 = np.ones(test_set.shape[0])  ##model1 = everyone lives

secret_test_set = test_set
secret_test_guess = np.zeros(secret_test_set.shape[0])

efficiency = Valid(test_truth, test_guess).value # closer to 1 the better.
print(efficiency)

efficiency2 = Valid(test_truth, test_guess2).value
print(efficiency2)

model = Arflou(train_set, train_result)
efficiency3 = Valid(test_truth, model.guess(test_set=test_set)).value
print(efficiency3)


