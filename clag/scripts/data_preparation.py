import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder

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
        self.process_age()
        self.process_sex()
        self.process_passenger_class()

    def process_age(self):
        self.training_data.Age.fillna(25, inplace=True)

    def process_sex(self):
        self.training_data['Sex'] = self.training_data['Sex'].map({'male': 1, 'female': 0})

    def process_passenger_class(self):
        for class_number in [1, 2, 3]:
            column_name = "Class_" + str(class_number)
            self.training_data.loc[self.training_data['Pclass'] == class_number, column_name] = 1
            self.training_data.loc[self.training_data['Pclass'] != class_number, column_name] = 0

    def save_data(self):
        self.training_data.to_pickle(CLEANED_DATA_FILEPATH)



def main():

    passengers = PassengerData()
    print(passengers.training_data)


if __name__ == '__main__':
    main()
