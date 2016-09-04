__author__ = 'melchior'


import csv
import os
import numpy as np

my_data = np.genfromtxt(os.path.join('..', 'data', 'train.csv'), delimiter=',')

print(my_data)

columns = ['PassengerId', 'Survived', 'Pclass', 'Name', 'Sex', 'Age', 'SibSp', 'Parch', 'Ticket', 'Fare', 'Cabin', 'Embarked']

df = [my_data, columns]

print(df[0])

