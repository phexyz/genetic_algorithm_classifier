from data import *
import tensorflow as tf
import numpy as np
import os
import datetime

def get_weights_bias(parameters, layer_name):

    weights = tf.constant(parameters[layer_name + "_W"], dtype=tf.float32)
    bias = tf.constant(parameters[layer_name + "_b"], dtype=tf.float32)

    return weights, bias


def conv(input, layer_name, weights_bias, activation="relu", pool=False):

    with tf.variable_scope(layer_name):
        weights, bias = weights_bias

        conv_no_bias = tf.nn.conv2d(input=input, filter=weights, strides=[1, 1, 1, 1], padding="SAME", name="without_bias")
        conv_z = tf.nn.bias_add(conv_no_bias, bias, name="bias_add")
        conv_a = tf.nn.relu(conv_z, name='relu')

        if pool:
            print("pool")
            conv_pool = tf.layers.max_pooling2d(conv_a, pool_size=[2,2], strides=2)
            return conv_pool

    return conv_a


def dense(input, layer_name, weights_bias, activation='relu', output=False):

    print("dense layer {}".format(layer_name))

    with tf.variable_scope(layer_name):


        print("1")
        if output:
            print("2")
            shape_pre_layer = input.get_shape().as_list()
            print("3")
            weights = tf.get_variable(name="weights", shape=[shape_pre_layer[1], 5], dtype=tf.float32)
            print("4")
            bias = tf.get_variable(name="bias", shape=[5, ], dtype=tf.float32)
            print("5")
        else:
            weights, bias = weights_bias

        dense_no_bias = tf.matmul(input, weights)
        print("9")
        dense_z = tf.nn.bias_add(dense_no_bias, bias)
        print("10")
        if activation == None:
            print("11")
            return dense_z

        print("12")
        dense_a = tf.nn.relu(dense_z, 'relu')
        print("13")
    return dense_a


def load_weights():
    file = np.load("vgg16_weights.npz")
    keys = file.keys()

    for key in keys:
        print(file[key].shape)
        print(key)

    print(len(keys))
    return file


class Vgg16(object):

    def __init__(self):
        self.build_model(load_weights())

    def build_model(self, parameters):

        self.iterator = read_TFRecord()
        self.sess = tf.Session()

        self.batch = self.iterator.get_next()

        self.X = self.batch[0]
        self.conv1_1 = conv(tf.cast(self.X, dtype=tf.float32), "conv1_1", get_weights_bias(parameters, "conv1_1"))
        self.conv1_2 = conv(self.conv1_1, "conv1_2", get_weights_bias(parameters, "conv1_2"), pool=True)
        self.conv2_1 = conv(self.conv1_2, "conv2_1", get_weights_bias(parameters, "conv2_1"))
        self.conv2_2 = conv(self.conv2_1, "conv2_2", get_weights_bias(parameters, "conv2_2"), pool=True)
        self.conv3_1 = conv(self.conv2_2, "conv3_1", get_weights_bias(parameters, "conv3_1"))
        self.conv3_2 = conv(self.conv3_1, "conv3_2", get_weights_bias(parameters, "conv3_2"))
        self.conv3_3 = conv(self.conv3_2, "conv3_3", get_weights_bias(parameters, "conv3_3"), pool=True)
        self.conv4_1 = conv(self.conv3_3, "conv4_1", get_weights_bias(parameters, "conv4_1"))
        self.conv4_2 = conv(self.conv4_1, "conv4_2", get_weights_bias(parameters, "conv4_2"))
        self.conv4_3 = conv(self.conv4_2, "conv4_3", get_weights_bias(parameters, "conv4_3"), pool=True)
        self.conv5_1 = conv(self.conv4_3, "conv5_1", get_weights_bias(parameters, "conv5_1"))
        self.conv5_2 = conv(self.conv5_1, "conv5_2", get_weights_bias(parameters, "conv5_2"))
        self.conv5_3 = conv(self.conv5_2, "conv5_3", get_weights_bias(parameters, "conv5_3"), pool=True)
        self.conv5_3_flatten = tf.contrib.layers.flatten(self.conv5_3)
        self.fc6 = dense(self.conv5_3_flatten, "fc6", get_weights_bias(parameters, "fc6"))
        self.fc7 = dense(self.fc6, "fc7", get_weights_bias(parameters, "fc7"))
        self.fc8 = dense(self.fc7, "fc8", get_weights_bias(parameters, "fc8"), output=True)
        self.Y_hat = tf.nn.softmax(self.fc8, name="softmax_output")

        self.Y = self.batch[1]
        self.loss = tf.losses.softmax_cross_entropy(self.Y, self.Y_hat)

        self.add_summery()


    def add_summery(self):

        root_logdir = "tf_logs"
        log_dir = os.path.join(root_logdir, str(datetime.datetime.now()))
        self.loss_summary = tf.summary.scalar(name="loss", tensor=self.loss)

        self.file_writer = tf.summary.FileWriter(log_dir, graph=tf.get_default_graph())

    def save_mode(self):

        saver = tf.train.Saver()
        saver.save(self.sess, os.path.join(os.getcwd(), "model"))

    def load_model(self):

        saver = tf.train.Saver()
        saver.restore(self.sess, "model")

    def train(self):

        self.train_variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "fc8")
        self.train_op = tf.train.AdamOptimizer(0.01).minimize(self.loss, var_list=self.train_variables)

        self.sess.run(tf.global_variables_initializer())

        num_epochs = 1000
        for epoch in range(num_epochs):
            print("epoch {}".format(epoch))
            self.sess.run(self.iterator.initializer)
            num_iter = 0
            try:
                while True:
                    print(num_iter)
                    num_iter += 1
                    summary, _ = self.sess.run([self.loss_summary, self.train_op])
                    self.file_writer.add_summary(summary)
            except tf.errors.OutOfRangeError:
                pass

        self.save_mode()


def main(argv):

    model = Vgg16()
    model.load_model()
    model.train()


if __name__ == '__main__':
    tf.app.run()