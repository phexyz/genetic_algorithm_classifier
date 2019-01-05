import numpy as np

class Individual(object):

    def __init__(self, argmap, value=None):
        """
        Parameters
        ----------
        argmap should contains the following parameters:
            1. population_size
            2. initializer
            3. truncation_size
            4. mutation_power
            5. max_generation
            6. name_to_shape
        """

        if not value:
            self.value = {}
            self.initialize_value(argmap["name_to_shape"])
        else:
            self.value = value
        self.error = None

    def initialize_value(self, name_to_shape):

        for layer_name in name_to_shape.keys():
            self.value[layer_name] = self.xavier_init(name_to_shape[layer_name])


    def xavier_init(self, layer_shape, uniform=False):
        """This function helps initialize the weights and bias using xavier initialization"""

        # Shape will contain the all the shape of every layer including the input layer.
        # But we only need to initialize the hidden layers and output layer.
        num_input = sum(layer_shape[1:])
        num_output = layer_shape[0]

        if uniform:
            init_range = np.sqrt(6.0 / (num_input + num_output))
            return np.random.uniform(low=-init_range, high=init_range, size=layer_shape)
        else:
            stddev = np.sqrt(2.0 / (num_input + num_output))
            return np.random.normal(loc=0, scale=1, size=layer_shape)


    def mutate(self, argmap):
        """This function mutates this individual"""

        # Not really sure whether epislon should be constant over one layer or constant over
        # individuals or over a generation.

        new_value = {}
        for key in self.value.keys():
            epsilon = np.random.normal(0, 1, size=self.value[key].shape)
            new_value[key] = self.value[key] + epsilon
        child = Individual(argmap=argmap, value=new_value)
        return child


    def returnValue(self):

        return self.value


    def printValueShape(self):

        for layer_value in self.value:
            print layer_value.shape


    def print_value(self):

        print self.value
