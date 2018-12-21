import tensorflow as tf
import numpy as np
from individual import *

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
        """

        self.population = []
        self.initialize_population(argmap)

    def evolve(self, argmap):
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

        max_generation = argmap["max_generation"]
        population_size = argmap["population_size"]
        mutation_power = argmap["mutation_power"]
        truncation_size = argmap["truncation_size"]

        population = []
        elite = None
        for g in range(max_generation):
            # g stands for generation
            for i in range(population_size - 1):
                # i stands for individual
                if g == 0:
                    child = Individual(argmap)
                    population.append(child)
                else:
                    # select the top T individual
                    parent = population[np.random.randint(0, truncation_size)]
                    child = self.mutate_individual(parent)
                # evaluate individual
                self.evaluate_inidvidual(child)

            if g != 1:
                population.append(Individual(argmap))
            else:
                population.append(elite)

            # sort individual by error
            population.sort(key=lambda x:x.error)
            elite = population[0]








    def initialize_population(self, argmap):
        """This function initialize a population of individuals with the same
        dimension as sample_individual"""

        for i in range(argmap["population_size"]):
            self.population.append(Individual(argmap))


    def mutate_population(self, potential_parents):
        """This function mutates the population"""
        pass


    def mutate_individual(self, individual):
        """This function mutates one individual"""
        pass


    def evaluate_population(self):
        """This function evaluates the error of the population"""
        pass


    def evaluate_inidvidual(self, individual):
        """This function evaluates the error of one individual"""
        pass


    def select_population(self):
        """This function selects a new generation of individuals from the
        population"""
        pass


    def select_individual(self, individual):
        """This function determines whether an individual should be selected
        for reproduction."""
        pass


