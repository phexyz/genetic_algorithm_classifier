import tensorflow as tf
import numpy as np

file = np.load("vgg16_weights.npz")
for item in file:
    print item
