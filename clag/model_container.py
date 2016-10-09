import pandas as pd
import numpy as np

from sklearn.linear_model import Lasso

file_location = '../data/header_train.csv'

columns = ['PassengerId', 'Pclass', 'Name', 'Sex', 'Age', 'SibSp', 'Parch', 'Ticket', 'Fare', 'Cabin', 'Embarked']
survival_training_df = pd.read_csv('../data/genderclassmodel.csv')


class PassengerData(object):
    def __init__(self):
        self.training_data = pd.read_csv(file_location)

    def get_X(self, data):
        return data[self.features]

    def get_Y(self, data):
        return data['Survived']


    def split_indices(self, data, num_splits):
        split_length = int(len(data) / num_splits)
        split_indices = [split_num * split_length for split_num in range(num_splits)]
        split_indices.append(len(data))
        pairs_of_train_vad_indices = []
        max_index = split_indices[-1]
        for split_ind, split_start in enumerate(split_indices[:-1]):
            split_end = split_start + (num_splits - 1) * split_length
            training_indices = np.array(range(split_start, split_end))
            training_indices = np.mod(training_indices, max_index)
            validation_indices = np.array(list(set(range(max_index)).difference(training_indices)))
            pairs_of_train_vad_indices.append((training_indices, validation_indices))
        return pairs_of_train_vad_indices

    def cross_validation(self, data, model):
        split_indices = self.split_indices(data, 3)
        prediction_errors = []
        for training_indices, validation_indices in split_indices:
            model.fit(self.get_X(data.ix[training_indices]), self.get_Y(data.ix[training_indices]))
            predicted_values = self.predict(model, data.ix[validation_indices])
            prediction_errors.append(
                sum(np.abs(predicted_values - self.get_Y(data.ix[validation_indices]))) / len(validation_indices))
        return prediction_errors


class LassoModel(PassengerData):
    def __init__(self):
        super(LassoModel, self).__init__()
        self.features = ['Fare']
        self.model = None

    def train(self, data):
        errors = []
        list_of_parameters = np.linspace(10 ** (-3), 10 ** 2, num=6)
        for parameter in list_of_parameters:
            model = Lasso(alpha=parameter)
            prediction_errors = self.cross_validation(data, model)
            errors.append((parameter, np.mean(prediction_errors)))

        errors.sort(key= lambda x: x[1])
        print(errors)
        optimal_parameter = errors[0][0]
        self.model = Lasso(alpha=optimal_parameter)
        self.model.fit(self.get_X(data), self.get_Y(data))


    def predict(self, model, data):
        values = model.predict(data[self.features])
        values[values < 0.5] = 0
        values[values >= 0.5] = 1
        return values


def main():
    passengers = PassengerData()

    print(passengers.training_data)
    print(passengers.get_X())
    print(passengers.get_Y())


if __name__ == '__main__':
    main()
