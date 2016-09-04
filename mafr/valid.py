__author__ = 'melchior'

import numpy as np
class Valid():

        def __init__(self, test_truth, test_guess):

            self.test_truth = test_truth
            self.test_guess = test_guess

            self.value = 1 - np.mean(abs(test_truth - test_guess))

