import os
import pandas as pd
import numpy as np

file_location = '../../data/header_train.csv'
CLEANED_DATA_FILEPATH = '../../data/cleaned_train_data.pck'
columns = ['PassengerId', 'Pclass', 'Name', 'Sex', 'Age', 'SibSp', 'Parch', 'Ticket', 'Fare', 'Cabin', 'Embarked']
#survival_training_df = pd.read_csv('../../data/genderclassmodel.csv')


class PassengerData(object):
    def __init__(self):
        self.training_data = pd.read_csv(file_location)
        self.process_training_data()
        self.save_data()

    def process_training_data(self):
        self.training_data.Age.fillna(25, inplace=True)

    def save_data(self):
        self.training_data.to_pickle(CLEANED_DATA_FILEPATH)



def main():

    passengers = PassengerData()
    print(passengers.training_data)


if __name__ == '__main__':
    main()
