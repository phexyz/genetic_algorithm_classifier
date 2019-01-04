from data import mnist
from data import data
from utils import utils
import tensorflow as tf
import numpy as np
import os
import datetime


class Vgg16(object):
    """This is the Vgg16 that can train and test"""

    def __init__(self):
        self.sess = tf.Session()
        self.build_model(utils.load_weights())

        self.train_variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                                                scope="parameters")

        self.feed_dict = {}
        self.name_variable_dict = {}


        # creates tf.train.saver to load and save trained model
        # self.save_model()

    def initialize_variables(self):
        """This function initialize all variables"""
        self.sess.run(tf.global_variables_initializer())

    def parameter_name_shape(self):
        """This function return the name and the shape of the parameters"""
        name_to_shape = {}
        for var in tf.get_collection("parameters"):
            name_to_shape[var.name] = var.get_shape().as_list()

        return name_to_shape

    def build_iterator(self):
        """This function build an iterator from tfRecord"""
        # This section sets up tf.data.iterator to input data
        self.train_iterator = read_TFRecord("train")
        self.test_iterator = read_TFRecord("validation")

        self.pred_input = tf.placeholder(dtype=tf.float32)

        self.handle = tf.placeholder(tf.string, shape=[])
        self.iterator = tf.data.Iterator.from_string_handle(
            self.handle, self.train_iterator.output_types)

        # The iterator to be used depends on whose handle is used.
        # This is passed through feed_dict in tf.Session.run.
        self.test_iterator_handle = self.sess.run(self.test_iterator.string_handle())
        self.train_iterator_handle = self.sess.run(self.train_iterator.string_handle())

    def build_model(self, parameters):
        """This function constructs the skeleton of the model"""

        # This is helpful for running the script on Google Colab as it resets all the tensorflow variables
        # tf.reset_default_graph()
        # This is the input from the iterator

        parameters = np.load("./vgg16_weights.npz")
        self.X = tf.placeholder(dtype=tf.float32, shape=[1, 28, 28, 1], name="input")
        self.conv1_1 = utils.conv(tf.cast(self.X, dtype=tf.float32), "conv1_1", placeholder=True)
        self.fc6 = utils.dense(tf.contrib.layers.flatten(self.conv1_1), "fc6", placeholder=True)
        self.fc7 = utils.dense(self.fc6, "fc7", placeholder=True)
        self.fc8 = utils.dense(self.fc7, "fc8",placeholder=True)
        self.Y_hat = tf.nn.softmax(self.fc8, name="softmax_output")

        self.Y = tf.placeholder(dtype=tf.float32, shape=(1, 10), name="label")

        self.loss = tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(labels=self.Y,
                                logits=self.fc8))


    def initialize_feed_dict(self, value):
        """This function generates a feed_dict. The keys are the pointer to the
        variables with the name as the key in the value. The value is the value
        of the variable with such name."""

        self.feed_dict = {}
        self.name_variable_dict = {}
        for key in value.keys():
            placeholder = tf.get_default_graph().get_tensor_by_name(key)
            self.name_variable_dict[key] = placeholder
            self.feed_dict[placeholder] = value[key]


    def error_fn(self, value):
        """This is the error function that calculates the loss given the values of the parameters"""

        if not self.feed_dict or not self.name_variable_dict:
            self.initialize_feed_dict(value)
        else:
            # update the value of the corresponding placeholder in the feed_dict
            for key in value.keys():
                self.feed_dict[self.name_variable_dict[key]] = value[key]

        self.mnist = mnist.MNIST()
        X, Y = self.mnist.load_train_dataset()

        for key in self.feed_dict:
            print key
        loss = 0
        shape = X.shape
        for i in range(shape[0]):
            print "The cumulative loss of all images before this image is {}".format(loss)

            self.feed_dict[self.X] = [X[i]]
            self.feed_dict[self.Y] = [Y[i]]
            loss += self.sess.run(self.loss, feed_dict=self.feed_dict)

        return loss
