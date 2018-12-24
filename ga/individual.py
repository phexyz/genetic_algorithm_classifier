import numpy as np

class Individual(object):

    def __init__(self, argmap):
        """
        Parameters
        ----------
        argmap should contains the following parameters:
            1. population_size
            2. initializer
            3. error_function
            4. truncation_size
            5. mutation_power
            6. max_generation
        """

        self.value = {}
        self.error = None

    def addLayer(self, layer_shape, layer_name):

        self.value[layer_name] = self.xavier_init(layer_shape)


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

        for i in range(len(self.value)):
            epsilon = np.random.normal(0, 1, self.value[i].shape)
            self.value[i] += argmap["mutation_power"] * epsilon
        return self

    def evaluate(self, argmap):
        """This function evalueates this individual"""

        error_fn = argmap["error_fn"]
        self.error = error_fn(self.value)


    def returnValue(self):

        return self.value


    def printValueShape(self):

        for layer_value in self.value:
            print layer_value.shape


    def print_value(self):

        print self.value


argmap = {"population_size": 10,
        "initializer": "xavier",
        "error_fn":None,
        "truncation_size":5,
        "mutation_power":0.002,
        "max_generation":50}
ind = Individual(argmap)
child = ind.mutate(argmap)
child.print_value()
child.printValueShape()
layer_name = "conv1_1"
layer_shape = (10, 10, 10)
child.addLayer(layer_shape, layer_name)
value = child.returnValue()
print value.keys()
