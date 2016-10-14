__author__ = 'melchior'

import pandas as pd
import numpy as np
import os

from clag.scripts.data_preparation import CLEANED_DATA_FILEPATH

class BaseClassModel():

    def __init__(self):

        dir_path = os.path.dirname(os.path.realpath(__file__))
        os.chdir(dir_path)
        self.training_data = pd.read_pickle(CLEANED_DATA_FILEPATH)

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

    def resplit_indices(self, size, num_splits):

        list_of_training_indices = []
        step = int(size/num_splits)
        for i in range(num_splits):
            list_of_training_indices.append(range(i * step, (i+1) * step))

        pass

    def cross_validation(self, data, model):
        split_indices = self.split_indices(data, 3)
        # split_indices = self.resplit_indices(len(data), 3)
        prediction_errors = []
        for training_indices, validation_indices in split_indices:

            cross_validation_subset_input = self.get_X(self.features).ix[training_indices]
            cross_validation_subset_target = self.get_Y().ix[training_indices]
            model.fit(cross_validation_subset_input, cross_validation_subset_target)
            predicted_values = model.predict(cross_validation_subset_input)

            predicted_values[predicted_values < 0.5] = 0
            predicted_values[predicted_values >= 0.5] = 1

            prediction_errors.append(
                sum(np.abs(predicted_values - cross_validation_subset_target)) / len(validation_indices))
        return prediction_errors

    def get_X(self, features):
        return self.training_data[features]

    def get_Y(self):
        return self.training_data['Survived']