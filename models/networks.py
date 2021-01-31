import tensorflow as tf
import numpy as np


def conv_layer(input_data, conv_filter, is_relu=False, is_scaling=False):
    """
    Parameters
    ----------
    x : input data
    conv_filter : weights of the filter
    is_relu : applies  ReLU activation function
    is_scaling : Scales the output

    """

    W = tf.get_variable('W', shape=conv_filter, initializer=tf.random_normal_initializer(0, 0.05))
    x = tf.nn.conv2d(input_data, W, strides=[1, 1, 1, 1], padding='SAME')

    if (is_relu):
        x = tf.nn.relu(x)

    if (is_scaling):
        scalar = tf.constant(0.1, dtype=tf.float32)
        x = tf.multiply(scalar, x)

    return x


def ResNet(input_data, nb_res_blocks):
    """

    Parameters
    ----------
    input_data : nrow x ncol x 2. Regularizer Input
    nb_res_blocks : default is 15.

    conv_filters : dictionary containing size of the convolutional filters applied in the ResNet
    intermediate outputs : dictionary containing intermediate outputs of the ResNet

    Returns
    -------
    nw_output : nrow x ncol x 2 . Regularizer output

    """

    conv_filters = dict([('w1', (3, 3, 2, 64)), ('w2', (3, 3, 64, 64)), ('w3', (3, 3, 64, 2))])
    intermediate_outputs = {}

    with tf.variable_scope('FirstLayer'):
        intermediate_outputs['layer0'] = conv_layer(input_data, conv_filters['w1'], is_relu=False, is_scaling=False)

    for i in np.arange(1, nb_res_blocks + 1):
        with tf.variable_scope('ResBlock' + str(i)):
            conv_layer1 = conv_layer(intermediate_outputs['layer' + str(i - 1)], conv_filters['w2'], is_relu=True, is_scaling=False)
            conv_layer2 = conv_layer(conv_layer1, conv_filters['w2'], is_relu=False, is_scaling=True)

            intermediate_outputs['layer' + str(i)] = conv_layer2 + intermediate_outputs['layer' + str(i - 1)]

    with tf.variable_scope('LastLayer'):
        rb_output = conv_layer(intermediate_outputs['layer' + str(i)], conv_filters['w2'], is_relu=False, is_scaling=False)

    with tf.variable_scope('Residual'):
        temp_output = rb_output + intermediate_outputs['layer0']
        nw_output = conv_layer(temp_output, conv_filters['w3'], is_relu=False, is_scaling=False)

    return nw_output


def mu_param():
    """
    Penalty parameter used in DC units, x = (E^h E + \mu I)^-1 (E^h y + \mu * z)
    """

    with tf.variable_scope(tf.get_variable_scope(), reuse=tf.AUTO_REUSE):
        mu = tf.get_variable(name='mu', dtype=tf.float32, initializer=.05)

    return mu
