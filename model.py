from data import *
import tensorflow as tf
import numpy as np
import os
import datetime

def conv(input, weights_name, bias_name, parameters, activation="relu", pool=False):

    with tf.variable_scope(weights_name[:-2]):
        weights = tf.constant(parameters[weights_name], dtype=tf.float16)
        bias = tf.constant(parameters[bias_name], dtype=tf.float16)

        conv_no_bias = tf.nn.conv2d(input=input, filter=weights, strides=[0, 1, 1, 0], padding="SAME", name="without_bias")
        conv_z = tf.nn.bias_add(conv_no_bias, bias, name="bias_add")
        conv_a = tf.nn.relu(conv_z, name='relu')

        if pool:
            print("pool")
            print(conv_a.get_shape().as_list())
            conv_pool = tf.layers.max_pooling2d(conv_a, pool_size=[2,2], strides=2)
            return conv_pool


    return conv_a

def dense(input, weights_name,  bias_name, parameters, activation='relu', output=False):

    print("dense layer {}".format(weights_name))

    with tf.variable_scope(weights_name[:-2]):

        if output:
            shape_pre_layer = input.get_shape().as_list()
            weights = tf.get_variable(name="weights", shape=[shape_pre_layer[1], 5], dtype=tf.float16)
            bias = tf.get_variable(name="bias", shape=[5, ], dtype=tf.float16)
        else:
            weights = tf.constant(parameters[weights_name], dtype=tf.float16)
            bias = tf.constant(parameters[bias_name], dtype=tf.float16)

        dense_no_bias = tf.matmul(input, weights)
        dense_z = tf.nn.bias_add(dense_no_bias, bias)

        if activation == None:
            return  dense_z


        dense_a = tf.nn.relu(dense_z, 'relu')

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

        self.sess = tf.Session()

        self.iterator = read_TFRecord()

        self.X = tf.placeholder(dtype=tf.float16, shape=[None, 224, 224, 3], name="input")
        self.conv1_1 = conv(self.X, "conv1_1_W", bias_name="conv1_1_b", parameters=parameters)
        self.conv1_2 = conv(self.conv1_1, "conv1_2_W", bias_name="conv1_2_b", parameters=parameters, pool=True)
        self.conv2_1 = conv(self.conv1_2, "conv2_1_W", bias_name="conv2_1_b", parameters=parameters)
        self.conv2_2 = conv(self.conv2_1, "conv2_2_W", bias_name="conv2_2_b", parameters=parameters, pool=True)
        self.conv3_1 = conv(self.conv2_2, "conv3_1_W", bias_name="conv3_1_b", parameters=parameters)
        self.conv3_2 = conv(self.conv3_1, "conv3_2_W", bias_name="conv3_2_b", parameters=parameters)
        self.conv3_3 = conv(self.conv3_2, "conv3_3_W", bias_name="conv3_3_b", parameters=parameters, pool=True)
        self.conv4_1 = conv(self.conv3_3, "conv4_1_W", bias_name="conv4_1_b", parameters=parameters)
        self.conv4_2 = conv(self.conv4_1, "conv4_2_W", bias_name="conv4_2_b", parameters=parameters)
        self.conv4_3 = conv(self.conv4_2, "conv4_3_W", bias_name="conv4_3_b", parameters=parameters, pool=True)
        self.conv5_1 = conv(self.conv4_3, "conv5_1_W", bias_name="conv5_1_b", parameters=parameters)
        self.conv5_2 = conv(self.conv5_1, "conv5_2_W", bias_name="conv5_2_b", parameters=parameters)
        self.conv5_3 = conv(self.conv5_2, "conv5_3_W", bias_name="conv5_3_b", parameters=parameters, pool=True)
        self.conv5_3_flatten = tf.contrib.layers.flatten(self.conv5_3)
        self.fc6 = dense(self.conv5_3_flatten, "fc6_W", bias_name="fc7_b", parameters=parameters)
        self.fc7 = dense(self.fc6, "fc7_W", bias_name="fc7_b", parameters=parameters)
        self.fc8 = dense(self.fc7, "fc8_W", bias_name="fc8_b", parameters=parameters, output=True)
        self.Y_hat = tf.nn.softmax(self.fc8, name="softmax_output")

        self.loss = tf.losses.softmax_cross_entropy


    def add_summery(self):

        root_logdir = "tf_logs"
        log_dir = os.path.join(root_logdir, str(datetime.datetime.now()))
        self.Y_hat_summary = tf.summary.tensor_summary(name="Y_hat_summary", tensor=self.Y_hat)
        self.file_write = tf.summary.FileWriter(log_dir, graph=tf.get_default_graph())

    def save_mode(self):

        saver = tf.train.Saver()
        saver.save(self.sess, os.path.join(os.getcwd(), "model"))

    def train(self):

        batch_X, batch_Y = self.iterator.get_next()

        self.train_variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "fc8")
        self.train_op = tf.train.AdamOptimizer(0.01).minimize(self.loss(batch_Y, self.Y_hat))

        self.sess.run(tf.global_variables_initializer())
        for iter in range(1000):
            print("iteration {}".format(iter))
            with self.sess as sess:
                sess.run(self.train_op, feed_dict={self.X : batch_X})



def main(argv):

    model = Vgg16()
    model.train()


if __name__ == '__main__':
    tf.app.run()