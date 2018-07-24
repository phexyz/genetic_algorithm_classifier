import tensorflow as tf
import numpy as np

def xavier_init(shape, name='', uniform=True):
    """This function helps initialize the weights and bias using xavier initialization"""

    num_input = sum(shape[:-1])
    num_output = shape[-1]

    if uniform:
        init_range = tf.sqrt(6.0 / (num_input + num_output))
        init_value = tf.random_uniform(shape=shape, minval=-init_range, maxval=init_range)
    else:
        stddev = tf.sqrt(3.0 / (num_input + num_output))
        init_value = tf.truncated_normal(shape=shape, stddev=stddev)

    return tf.Variable(init_value, name=name, trainable=True)


def get_weights_bias(parameters, layer_name, output=False):
    """This function gets the weights and bias for a given layer whether it be initialization or obtaining weights
        from trained model"""

    # fc stands for fully connected layer
    if "fc" in layer_name:

        if output:
            weights = xavier_init(shape=[parameters[layer_name + "_W"].shape[0], 5], name="{}/weights".format(layer_name))
            bias = xavier_init(shape=[5, ], name="{}/bias".format(layer_name))

        else:
            weights = tf.Variable(parameters[layer_name + "_W"], name="{}/weights".format(layer_name))
            bias = tf.Variable(parameters[layer_name + "_b"], name="{}/bias".format(layer_name))

    else:
        weights = tf.constant(parameters[layer_name + "_W"], dtype=tf.float32, name="{}/weights".format(layer_name))
        bias = tf.constant(parameters[layer_name + "_b"], dtype=tf.float32, name="{}/bias".format(layer_name))

    return weights, bias


def conv(input, layer_name, weights_bias, activation="relu", pool=False):
    """This function is used to build a layer of convolution including the pooling and pooling layer"""

    print("convolution layer {0} {1} pooling layer".format(layer_name, "with" if pool else "without"))

    with tf.variable_scope(layer_name):
        weights, bias = weights_bias

        conv_no_bias = tf.nn.conv2d(input=input, filter=weights, strides=[1, 1, 1, 1], padding="SAME",
                                    name="without_bias")
        conv_z = tf.nn.bias_add(conv_no_bias, bias, name="bias_add")
        conv_a = tf.nn.relu(conv_z, name='relu')

        if pool:
            conv_pool = tf.layers.max_pooling2d(conv_a, pool_size=[2, 2], strides=2)
            return conv_pool

        return conv_a


def dense(input, layer_name, weights_bias, activation='relu'):
    """This function is used to build a fully connected layer and pooling layer"""

    print("dense layer {}".format(layer_name))

    with tf.variable_scope(layer_name):
        weights, bias = weights_bias

        dense_no_bias = tf.matmul(input, weights)
        dense_z = tf.nn.bias_add(dense_no_bias, bias)

        if activation == None:
            return dense_z

        dense_a = tf.nn.relu(dense_z, 'relu')
        return dense_a


def load_weights():
    """This function loads the weights from vgg16"""

    file = np.load("vgg16_weights.npz")
    return file


def mnist_iterator():
    """This function creates a mnist iterator that goes through mnist dataset"""

    # obtain dataset
    train, test = tf.keras.datasets.mnist.load_data()
    mnist_x, mnist_y = train

    # create tf.dataset
    dataset_x = tf.data.Dataset.from_tensor_slices(mnist_x)
    dataset_y = tf.data.Dataset.from_tensor_slices(mnist_y)

    dataset_x = dataset_x.batch(100)
    dataset_y = dataset_y.batch(100)

    # create iterator
    iterator_y = dataset_y.make_initializable_iterator()
    iterator_x = dataset_x.make_initializable_iterator()

    return iterator_x, iterator_y
