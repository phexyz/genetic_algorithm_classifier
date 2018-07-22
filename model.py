from data import *
import tensorflow as tf
import numpy as np
import os
import datetime


def xavier_init(shape, name='', uniform=True):
    num_input = sum(shape[:-1])
    num_output = shape[-1]

    if uniform:
        init_range = tf.sqrt(6.0 / (num_input + num_output))
        init_value = tf.random_uniform(shape=shape, minval=-init_range, maxval=init_range)
    else:
        stddev = tf.sqrt(3.0 / (num_input + num_output))
        init_value = tf.truncated_normal(shape=shape, stddev=stddev)

    return tf.Variable(init_value, name=name)


def get_weights_bias(parameters, layer_name):
    if "fc" in layer_name:

        if "8" in layer_name:
            weights = xavier_init(shape=[parameters[layer_name + "_W"].shape[0], 5], name="weights")
            bias = xavier_init(shape=[5, ], name="bias")
        else:
            weights = tf.Variable(parameters[layer_name + "_W"], name="{}/weights".format(layer_name))
            bias = tf.Variable(parameters[layer_name + "_b"], name="{}/bias".format(layer_name))

    else:
        weights = tf.constant(parameters[layer_name + "_W"], dtype=tf.float32, name="{}/weights".format(layer_name))
        bias = tf.constant(parameters[layer_name + "_b"], dtype=tf.float32, name="{}/bias".format(layer_name))

    return weights, bias


def conv(input, layer_name, weights_bias, activation="relu", pool=False):
    with tf.variable_scope(layer_name):
        weights, bias = weights_bias

        conv_no_bias = tf.nn.conv2d(input=input, filter=weights, strides=[1, 1, 1, 1], padding="SAME",
                                    name="without_bias")
        conv_z = tf.nn.bias_add(conv_no_bias, bias, name="bias_add")
        conv_a = tf.nn.relu(conv_z, name='relu')

        if pool:
            print("pool")
            conv_pool = tf.layers.max_pooling2d(conv_a, pool_size=[2, 2], strides=2)
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
            weights = xavier_init(shape=[shape_pre_layer[1], 5], name="weights")
            # weights = tf.get_variable(name="weights", shape=[shape_pre_layer[1], 5], dtype=tf.float32,
            #                           initializer=tf.initializers.)
            print("4")
            bias = xavier_init(shape=[5, ], name="bias")
            # bias = tf.get_variable(name="bias", shape=[5, ], dtype=tf.float32)
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

        self.train_iterator = read_TFRecord("train")
        self.test_iterator = read_TFRecord("validation")

        self.handle = tf.placeholder(tf.string, shape=[])
        self.iterator = tf.data.Iterator.from_string_handle(
            self.handle, self.train_iterator.output_types)
        self.sess = tf.Session()

        self.test_iterator_handle = self.sess.run(self.test_iterator.string_handle())
        self.train_iterator_handle = self.sess.run(self.train_iterator.string_handle())

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

        self.train_variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="fc8") + \
                               tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="fc7") + \
                               tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="fc6")

        self.save_mode()
        self.load_model()

    def add_summery(self):

        root_logdir = "tf_logs"
        log_dir = os.path.join(root_logdir, str(datetime.datetime.now()))
        self.loss_summary = tf.summary.scalar(name="loss", tensor=self.loss)
        self.file_writer = tf.summary.FileWriter(log_dir, graph=tf.get_default_graph())

    def save_mode(self):

        if not os.path.exists("model"):
            os.makedirs("model")
        self.saver = tf.train.Saver(var_list=self.train_variables)

    def load_model(self):

        if os.path.exists("model/"):
            self.saver.restore(self.sess, save_path="model/model")
        else:
            print("model doesn't exist")

    def train(self):

        for var in self.train_variables:
            print(var.name)

        self.train_op = tf.train.AdamOptimizer(0.00075).minimize(self.loss, var_list=self.train_variables)

        self.sess.run(tf.global_variables_initializer())
        self.add_summery()

        num_epochs = 1000
        for epoch in range(num_epochs):
            print("epoch {}".format(epoch))
            self.sess.run(self.train_iterator.initializer)

            num_iter = 0
            try:
                while True:
                    num_iter += 1
                    loss, summary, _ = self.sess.run([self.loss, self.loss_summary, self.train_op],
                                                     feed_dict={self.handle : self.train_iterator_handle})
                    self.file_writer.add_summary(summary)
                    print num_iter,

                    # download_folder("model")

            except tf.errors.OutOfRangeError:
                pass

            if epoch % 50 == 0:

                self.saver.save(self.sess, os.path.join(os.getcwd(), "model/"))

                # upload_folder_to_s3("model")

    def test(self):

        self.batch_correct = tf.reduce_sum(tf.cast(tf.equal(tf.argmax(self.Y, -1), tf.argmax(self.Y_hat, -1)), tf.float32))
        accuracy = 0
        counter = 0

        self.sess.run([tf.global_variables_initializer(), self.test_iterator.initializer])
        try:
            while True:
                batch_accuracy = self.sess.run(self.batch_correct, feed_dict={self.handle : self.test_iterator_handle})
                accuracy += batch_accuracy
                counter += 1
                print(counter)


        except tf.errors.OutOfRangeError:
            pass

        print("coutner", counter)
        print("accuracy", accuracy)

    def predict(self, input):

        self.X = tf.placeholder(dtype=tf.float32, shape=[None, 224, 224, 3], name="input")
        self.prediction = tf.argmax(self.Y_hat)

        self.sess.run(tf.global_variables_initializer())
        prediction = self.sess.run(self.prediction, self.X)

        return prediction



def main(argv):

    model = Vgg16()
    model.test()


if __name__ == '__main__':
    tf.app.run()
