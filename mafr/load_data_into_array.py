__author__ = 'melchior'


import csv
import os

with open(os.path.join('..', 'data', 'train.csv'), 'rb') as csvfile:
    myreader = csv.reader(csvfile, delimiter=' ', quotechar='|')


print(myreader)