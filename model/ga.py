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
            next_generation = []
            for i in range(population_size - 1):
                # i stands for individual
                print "individual {0}, population {1}".format(i, len(population))
                if g == 0:
                    child = Individual(self.argmap)
                else:
                    # select the top T individual
                    parent = population[np.random.randint(0, truncation_size)]
                    child = parent.mutate(self.argmap)
                # evaluate individual
                self.evaluate(child)
                next_generation.append(child)

            if g == 0:
                next_generation.append(Individual(self.argmap))
            else:
                next_generation.append(elite)

            print "Generation {0}, Error{1}".format(g, child.error)
            # sort individual by error
            next_generation.sort(key=lambda individual: individual.error)
            elite = next_generation[0]
            population = next_generation


    def evaluate(self, individual):
        """This function evaluates the individual """

        print "-----------evalueate this individual-----------"
        individual.error = self.model.error_fn(individual.value)
