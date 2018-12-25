from data import *
from s3_functions import *
from utils import *
import tensorflow as tf
import os
import datetime

# This is helpful for running the script on Google Colab as it resets all the tensorflow variables
tf.reset_default_graph()

class Vgg16(object):
    """This is the Vgg16 that can train and test"""

    def __init__(self):
        self.sess = tf.Session()
        self.build_iterator()
        self.build_model(load_weights())

        self.train_variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                                                scope="parameters")

        self.feed_dict = {}
        self.name_variable_dict = {}


        # creates tf.train.saver to load and save trained model
        # self.save_model()

    def initialize_variables(self):
        """This function initialize all variables"""
        self.sess.run([tf.global_variables_initializer(), self.train_iterator.initializer])

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


        # This is the input from the iterator
        self.batch = self.iterator.get_next()

        self.X = self.batch[0]
        self.conv1_1 = conv(tf.cast(self.X, dtype=tf.float32), "conv1_1")
        self.conv1_2 = conv(self.conv1_1, "conv1_2", pool=True)
        self.conv2_1 = conv(self.conv1_2, "conv2_1")
        self.conv2_2 = conv(self.conv2_1, "conv2_2", pool=True)
        self.conv3_1 = conv(self.conv2_2, "conv3_1")
        self.conv3_2 = conv(self.conv3_1, "conv3_2")
        self.conv3_3 = conv(self.conv3_2, "conv3_3", pool=True)
        self.conv4_1 = conv(self.conv3_3, "conv4_1")
        self.conv4_2 = conv(self.conv4_1, "conv4_2")
        self.conv4_3 = conv(self.conv4_2, "conv4_3", pool=True)
        self.conv5_1 = conv(self.conv4_3, "conv5_1")
        self.conv5_2 = conv(self.conv5_1, "conv5_2")
        self.conv5_3 = conv(self.conv5_2, "conv5_3", pool=True)
        self.conv5_3_flatten = tf.contrib.layers.flatten(self.conv5_3)
        self.fc6 = dense(self.conv5_3_flatten, "fc6")
        self.fc7 = dense(self.fc6, "fc7")
        self.fc8 = dense(self.fc7, "fc8")
        self.Y_hat = tf.nn.softmax(self.fc8, name="softmax_output")

        self.Y = self.batch[1]

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

        self.feed_dict[self.handle] = self.train_iterator_handle


    def error_fn(self, value):
        if not self.feed_dict or not self.name_variable_dict:
            self.initialize_feed_dict(value)
        else:
            # update the value of the corresponding placeholder in the feed_dict
            for key in value.keys():
                self.feed_dict[self.name_variable_dict[key]] = value[key]

        return self.sess.run(self.loss, feed_dict=self.feed_dict)

    def add_summery(self, root_logdir):
        """This function adds summary"""

        log_dir = os.path.join(root_logdir, str(datetime.datetime.now()))
        self.loss_summary = tf.summary.scalar(name="loss", tensor=self.loss)
        self.file_writer = tf.summary.FileWriter(log_dir, graph=tf.get_default_graph())

    def save_model(self):
        """This function creates a saver"""

        if not os.path.exists("model"):
            os.makedirs("model")
        self.saver = tf.train.Saver(var_list=self.train_variables)

    def load_model(self, checkpoint_dir):
        """This function loads a trained model"""

        if os.path.exists(checkpoint_dir):
            self.saver.restore(self.sess, save_path="{}/model".format(checkpoint_dir))
        else:
            print("Model doesn't exist. Train new model")

    def train(self, learning_rate, epoch_num, checkpoint_dir, summary_dir):
        """This function trains the model"""

        print("Here are the variables to be trained")
        for var in self.train_variables:
            print(var.name)

        self.train_op = tf.train.AdamOptimizer(learning_rate).minimize(self.loss, var_list=self.train_variables)

        self.sess.run([tf.global_variables_initializer()])
        self.add_summery(summary_dir)
        self.load_model(checkpoint_dir)

        for epoch in range(epoch_num):
            print("epoch {} losses".format(epoch))

            # Initialize the iterator at the beginning of each epoch
            self.sess.run([self.train_iterator.initializer])
            try:
                while True:
                    loss, summary, _ = self.sess.run([self.loss, self.loss_summary, self.train_op],
                                                     feed_dict={self.handle: self.train_iterator_handle})
                    self.file_writer.add_summary(summary)
                    print loss,

            except tf.errors.OutOfRangeError:
                pass

            if epoch % 50 == 0 and epoch != 0:
                self.saver.save(self.sess, os.path.join(os.getcwd(), "{}/model".format(checkpoint_dir)))

    def test(self, checkpoint_dir):
        """This function goes through the test set"""

        self.batch_correct = tf.reduce_sum(
            tf.cast(tf.equal(tf.argmax(self.Y, -1), tf.argmax(self.Y_hat, -1)), tf.float32))
        accuracy = 0
        counter = 0

        self.sess.run([tf.global_variables_initializer(), self.test_iterator.initializer])
        self.load_model(checkpoint_dir)

        try:
            while True:
                batch_accuracy, Y, Y_hat = self.sess.run([self.batch_correct, self.Y, self.Y_hat],
                                                         feed_dict={self.handle: self.test_iterator_handle})
                print("batch accuracy", batch_accuracy, "counter", counter)

                accuracy += batch_accuracy
                counter += 1

        except tf.errors.OutOfRangeError:
            pass

        print("coutner", counter)
        print("accuracy", accuracy)

    def predict(self, input, checkpoint_dir):
        """This function predicts a single image"""

        inputs = os.listdir(input) if os.path.isdir(input) else [input]

        from PIL import Image
        for input in list(map(lambda x: os.path.join(input, x), inputs)):

            orig_img = Image.open(input)
            resize_img = orig_img.resize((244, 244))
            img_array = np.array(resize_img)

            dataset_x = tf.data.Dataset.from_tensor_slices(tf.expand_dims(img_array, 0))
            dataset_y = tf.data.Dataset.from_tensor_slices(tf.cast([0, 0, 0, 0, 0], tf.float32))
            dataset = tf.data.Dataset.zip((dataset_x, dataset_y))
            dataset = dataset.batch(1)
            iterator = dataset.make_one_shot_iterator()
            iterator_handle = self.sess.run(iterator.string_handle())


            self.sess.run(tf.global_variables_initializer())
            self.load_model(checkpoint_dir)
            onehot_prediction = self.sess.run([self.Y_hat], feed_dict={self.handle : iterator_handle})
            prediction = np.argmax(onehot_prediction)


            prediction_map = {0:"daisy",
                              1:"dandelion",
                              2:"rose",
                              3:"sunflower",
                              4:"tulip"}

            print("VGG16 thinks this is a {}".format(prediction_map[prediction]))
            resize_img.show()
