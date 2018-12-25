import tensorflow as tf
import numpy as np
from individual import *
from model import *

class GAWorker(object):

    def __init__(self, argmap):
        """
        Parameters
        ----------
        argmap should contains the following parameters:
            1. shape (shape of an individual)
            2. population_size
            3. initializer
            4. error_function
            5. truncation_size
            6. mutation_power
            7. max_generation
            8. standard
        """
        self.population = []
        self.model = Vgg16()
        self.model.initialize_variables()
        self.argmap = argmap
        self.argmap["name_to_shape"] = self.model.parameter_name_shape()

    def evolve(self):
        """
        Parameters
        ----------
        argmap should contains the following parameters:
            1. shape (shape of an individual)
            2. population_size
            3. initializer
            4. error_function
            5. truncation_size
            6. mutation_power
            7. max_generation
            8. standard
        """

        max_generation = self.argmap["max_generation"]
        population_size = self.argmap["population_size"]
        mutation_power = self.argmap["mutation_power"]
        truncation_size = self.argmap["truncation_size"]

        population = []
        elite = None
        for g in range(max_generation):
            # g stands for generation
            print "generation {}".format(g)
            for i in range(population_size - 1):
                # i stands for individual
                print "individual {0}, population {1}".format(i, len(population))
                if g == 0:
                    child = Individual(self.argmap)
                    population.append(child)
                else:
                    # select the top T individual
                    parent = population[np.random.randint(0, truncation_size)]
                    child = parent.mutate(argmap)
                # evaluate individual
                self.evaluate(child)

            if g != 1:
                population.append(Individual(self.argmap))
            else:
                population.append(elite)

            print "Generation {0}, Error{1}".format(g, child.error)
            # sort individual by error
            population.sort(key=lambda x: x.error)
            elite = population[0]


    def evaluate(self, individual):
        """This function evaluates the individual """

        individual.error = self.model.error_fn(individual.value)

argmap = {"population_size": 2,
        "initializer": "xavier",
        "error_fn":None,
        "truncation_size":2,
        "mutation_power":0.002,
        "max_generation":50}
worker = GAWorker(argmap)
worker.evolve()
