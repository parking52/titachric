from clag.models.BaseClassModel import BaseClassModel


class Mendie(BaseClassModel):

    # def __init__(self):
    #     super(BaseClassModel, self).__init__()
    #     self.features = None
    #     self.model = None
    #     self.optimal_parameter = None

    def train(self):
        pass
        prediction = self.predict(self.training_data)
        wrongs_percentage = abs(self.training_data.Survived - prediction).sum() / self.training_data.Survived.__len__()
        print('In mendie womanlive, the error percentage is ' + str(wrongs_percentage))
        print('Hence the accuracy percentage is ' + str(1 - wrongs_percentage))

    def predict(self, input_data):
        values = abs(input_data.Sex-1)
        return values