import tensorflow as tf
from model import ga

def main():
    argmap = {"population_size": 4,
            "initializer": "xavier",
            "error_fn":None,
            "truncation_size":2,
            "mutation_power":0.002,
            "max_generation":50}
    worker = ga.GAWorker(argmap)
    worker.evolve()

if __name__ == '__main__':
    main()
