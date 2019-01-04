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


def getWeightsBiasShape(layer_name):
    """This function gets the weights and bias for a given layer whether it be
    initialization or obtaining weights from trained model"""

    parameters = {"conv1_1_W" : (3, 3, 3, 64)
            , "conv1_1_b" : (64,)
            , "conv1_2_W" : (3, 3, 64, 64)
            , "conv1_2_b" : (64,)
            , "conv2_1_b" : (128,)
            , "conv2_1_W" : (3, 3, 64, 128)
            , "conv2_2_W" : (3, 3, 128, 128)
            , "conv2_2_b" : (128,)
            , "conv3_1_W" : (3, 3, 128, 256)
            , "conv3_1_b" : (256,)
            , "conv3_2_W" : (3, 3, 256, 256)
            , "conv3_2_b" : (256,)
            , "conv3_3_W" : (3, 3, 256, 256)
            , "conv3_3_b" : (256,)
            , "conv4_1_W" : (3, 3, 256, 512)
            , "conv4_1_b" : (512,)
            , "conv4_2_W" : (3, 3, 512, 512)
            , "conv4_2_b" : (512,)
            , "conv4_3_W" : (3, 3, 512, 512)
            , "conv4_3_b" : (512,)
            , "conv5_1_W" : (3, 3, 512, 512)
            , "conv5_1_b" : (512,)
            , "conv5_2_W" : (3, 3, 512, 512)
            , "conv5_2_b" : (512,)
            , "conv5_3_W" : (3, 3, 128, 512)
            , "conv5_3_b" : (512,)
            , "fc6_W" : (25088, 4096)
            , "fc6_b" : (4096,)
            , "fc7_W" : (4096, 4096)
            , "fc7_b" : (4096,)
            , "fc8_W" : (4096, 1000)
            , "fc8_b" : (1000,)
            }

    weightsShape = parameters[layer_name + "_W"]
    biasShape = parameters[layer_name + "_b"]

    # fc stands for fully connected layer
    if layer_name == "fc8":
        weightsShape = (parameters[layer_name + "_W"][0], 5)
        biasShape = (5, )

    return weightsShape, biasShape


def conv(input, layer_name, activation="relu", pool=False, placeholder=False, parameters=None):
    """This function is used to build a layer of convolution including
    the pooling and pooling layer"""

    print("convolution layer {0} {1} pooling layer".format(layer_name,
        "with" if pool else "without"))

    with tf.variable_scope(layer_name):

        if placeholder:
            weightsShape, biasShape = getWeightsBiasShape(layer_name)
            weights = tf.placeholder(dtype=tf.float32, shape=weightsShape,
                                    name="weights")
            bias = tf.placeholder(dtype=tf.float32, shape=biasShape,
                                    name="bias")
            tf.add_to_collection(name="parameters", value=weights)
            tf.add_to_collection(name="parameters", value=bias)
        else:
            weights = parameters[layer_name + "_W"]
            bias = parameters[layer_name + "_b"]

        conv_no_bias = tf.nn.conv2d(input=input, filter=weights,
                                    strides=[1, 1, 1, 1], padding="SAME",
                                    name="without_bias")
        conv_z = tf.nn.bias_add(conv_no_bias, bias, name="bias_add")
        conv_a = tf.nn.relu(conv_z, name='relu')

        if pool:
            conv_pool = tf.layers.max_pooling2d(conv_a, pool_size=[2, 2], strides=2)
            return conv_pool

        return conv_a


def dense(input, layer_name, activation='relu', placeholder=False, parameters=None):
    """This function is used to build a fully connected layer and pooling layer"""

    print("dense layer {}".format(layer_name))

    with tf.variable_scope(layer_name):

        if placeholder:
            weightsShape, biasShape = getWeightsBiasShape(layer_name)
            weights = tf.placeholder(dtype=tf.float32, shape=weightsShape,
                                    name="weights")
            bias = tf.placeholder(dtype=tf.float32, shape=biasShape,
                                    name="bias")
            tf.add_to_collection(name="parameters", value=weights)
            tf.add_to_collection(name="parameters", value=bias)
        else:
            weights = parameters[layer_name + "_W"]
            bias = parameters[layer_name + "_b"]

        dense_no_bias = tf.matmul(input, weights)
        dense_z = tf.nn.bias_add(dense_no_bias, bias)

        if activation == None:
            return dense_z

        dense_a = tf.nn.relu(dense_z, 'relu')
        return dense_z


def load_weights():
    """This function loads the weights from vgg16"""

    return {"conv1_1_W" : (3, 3, 3, 64)
            , "conv1_1_b" : (64,)
            , "conv1_2_W" : (3, 3, 64, 64)
            , "conv1_2_b" : (64,)
            , "conv2_1_b" : (128,)
            , "conv2_1_W" : (3, 3, 64, 128)
            , "conv2_2_W" : (3, 3, 128, 128)
            , "conv2_2_b" : (128,)
            , "conv3_1_W" : (3, 3, 128, 256)
            , "conv3_1_b" : (256,)
            , "conv3_2_W" : (3, 3, 256, 256)
            , "conv3_2_b" : (256,)
            , "conv3_3_W" : (3, 3, 256, 256)
            , "conv3_3_b" : (256,)
            , "conv4_1_W" : (3, 3, 256, 512)
            , "conv4_1_b" : (512,)
            , "conv4_2_W" : (3, 3, 512, 512)
            , "conv4_2_b" : (512,)
            , "conv4_3_W" : (3, 3, 512, 512)
            , "conv4_3_b" : (512,)
            , "conv5_1_W" : (3, 3, 512, 512)
            , "conv5_1_b" : (512,)
            , "conv5_2_W" : (3, 3, 512, 512)
            , "conv5_2_b" : (512,)
            , "conv5_3_W" : (3, 3, 512, 512)
            , "conv5_3_b" : (512,)
            , "fc6_W" : (25088, 4096)
            , "fc6_b" : (4096,)
            , "fc7_W" : (4096, 4096)
            , "fc7_b" : (4096,)
            , "fc8_W" : (4096, 1000)
            , "fc8_b" : (1000,)
            }

def get_all_variables_with_name(var_name):
    """This function gets the variable with var_name"""

    return tf.get_default_graph().get_tensor_by_name(var_name + ":0")

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
