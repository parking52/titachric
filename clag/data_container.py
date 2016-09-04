import pandas as pd
import numpy as np

from sklearn.linear_model import Lasso

file_location = '../data/header_train.csv'

columns = ['PassengerId', 'Pclass', 'Name', 'Sex', 'Age', 'SibSp', 'Parch', 'Ticket', 'Fare', 'Cabin', 'Embarked']
survival_training_df = pd.read_csv('../data/genderclassmodel.csv')


class PassengerData(object):
    def __init__(self):
        #self.training_data = np.genfromtxt(os.path.join('..', 'data', 'train.csv'), delimiter=',')
        self.training_data = pd.read_csv(file_location)

    def get_X(self):
        return self.training_data[self.features]

    def get_Y(self):
        return self.training_data['Survived']


class LassoModel(PassengerData):
    def __init__(self):
        super(LassoModel, self).__init__()
        self.features = ['Fare']

    def train(self):
        self.model = Lasso()
        self.model.fit(self.get_X(), self.get_Y())

    def predict(self, test_data):
        return self.model.predict(test_data.ix[:20])





def main():
    passengers = PassengerData()
    passengers.training_data['Age']
    print(passengers.training_data)
    print(passengers.get_X())
    print(passengers.get_Y())


if __name__ == '__main__':
    main()

