import tensorflow as tf
from model import *

flags = tf.app.flags
flags.DEFINE_boolean("train", True, "True for training")
flags.DEFINE_boolean("test", False, "True for testing")
flags.DEFINE_integer("epoch", 1000, "Epoch to train [1000]")
flags.DEFINE_float("learning_rate", 0.000075, "Learning rate for training [0.000075]")
flags.DEFINE_string("checkpoint_dir", "model", "Checkpoint directory []")
flags.DEFINE_string("input_image_path", "", "input image name []")
FLAGS = flags.FLAGS


def main(_):

    model = Vgg16()

    if FLAGS.input_image_path == "":

        if FLAGS.test:
            model.test(FLAGS.checkpoint_dir)

        else:
            model.train(FLAGS.learning_rate, FLAGS.epoch, FLAGS.checkpoint_dir)


    else:

        model.predict(FLAGS.input_image_path, FLAGS.checkpoint_dir)


if __name__ == '__main__':
    tf.app.run()