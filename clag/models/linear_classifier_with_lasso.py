import pandas as pd
import numpy as np

from sklearn.linear_model import Lasso, LogisticRegression
from clag.models.BaseClassModel import BaseClassModel

list_of_parameters = np.logspace((-3),2, num = 6)
list_of_parameters = [0, 0.01, 0.2, 0.5, 1 ]



class LassoModel(BaseClassModel):
    def __init__(self):
        super(LassoModel, self).__init__()
        # self.features = ['Age', 'Fare', 'Sex', 'Class_1', 'Class_2', 'Class_3']
        self.features = ['Sex']
        self.model = None
        self.optimal_parameter = None

    def train(self):
        errors = []
        for parameter in list_of_parameters:
            model = Lasso(alpha=parameter)
            prediction_errors = self.cross_validation(self.training_data, model)
            errors.append((parameter, np.mean(prediction_errors)))
            print(str(model.coef_) + '    ' + str(parameter))

        errors.sort(key=lambda x: x[1])
        print(errors)

        optimal_parameter = errors[0][0]
        self.optimal_parameter = optimal_parameter
        self.model = Lasso(alpha=optimal_parameter)
        self.model.fit(self.get_X(self.features), self.get_Y())

    def predict(self, input_data):
        values = self.model.predict(input_data)
        values[values < 0.5] = 0
        values[values >= 0.5] = 1
        return values


class LogisticModel(BaseClassModel):
    def __init__(self):
        super(LogisticModel, self).__init__()
        self.features = ['Age', 'Fare']
        self.model = None
        self.optimal_parameter = None

    def train(self):
        errors = []
        model = LogisticModel()
        prediction_errors = self.cross_validation(self.training_data, model)
        errors.append((parameter, np.mean(prediction_errors)))

        errors.sort(key=lambda x: x[1])
        print(errors)
        optimal_parameter = errors[0][0]
        self.optimal_parameter = optimal_parameter
        self.model = Lasso(alpha=optimal_parameter)
        self.model.fit(self.get_X(self.features), self.get_Y())


if __name__ == '__main__':
    model = LassoModel()
