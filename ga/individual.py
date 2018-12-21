import tensorflow as tf
import numpy as np


class Individual(object):

    def __init__(self, argmap):
        """
        Parameters
        ----------
        argmap should contains the following parameters:
            1. shape (shape of an individual)
            2. population_size
            3. initializer
            4. error_function
        """

        self.value = self.init(argmap)
        self.error = None


    def init(self, argmap):
        """This function initialize the initial values of this individual"""

        shape = argmap["shape"]
        initializer = argmap["initializer"]
        if not initializer:
            return np.random.rand(shape)
