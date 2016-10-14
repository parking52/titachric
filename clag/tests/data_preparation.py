__author__ = 'melchior'

from clag.models.BaseClassModel import BaseClassModel

def test_reading_file():
    model = BaseClassModel()

    assert model.training_data.shape == (891, 12)