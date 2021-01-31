import tensorflow as tf
import math
import os
import parser_ops
import UnrollNet

parser = parser_ops.get_parser()
args = parser.parse_args()


def test_graph(directory):
    """
    This function creates a test graph for testing
    """

    tf.reset_default_graph()
    # %% placeholders for the unrolled network
    sens_mapsP = tf.placeholder(tf.complex64, shape=(None, args.ncoil_GLOB, args.nrow_GLOB, args.ncol_GLOB), name='sens_maps')
    trn_maskP = tf.placeholder(tf.complex64, shape=(None, args.nrow_GLOB, args.ncol_GLOB), name='trn_mask')
    loss_maskP = tf.placeholder(tf.complex64, shape=(None, args.nrow_GLOB, args.ncol_GLOB), name='loss_mask')
    nw_inputP = tf.placeholder(tf.float32, shape=(None, args.nrow_GLOB, args.ncol_GLOB, 2), name='nw_input')

    nw_output, nw_kspace_output, x0, all_intermediate_outputs, mu = \
               UnrollNet.UnrolledNet(nw_inputP, sens_mapsP, trn_maskP, loss_maskP).model

    # %% unrolled network outputs
    nw_output = tf.identity(nw_output, name='nw_output')
    nw_kspace_output = tf.identity(nw_kspace_output, name='nw_kspace_output')
    all_intermediate_outputs = tf.identity(all_intermediate_outputs, name='all_intermediate_outputs')
    x0 = tf.identity(x0, name='x0')
    mu = tf.identity(mu, name='mu')

    # %% saves computational graph for test
    saver = tf.train.Saver()
    sess_test_filename = os.path.join(directory, 'model_test')
    with tf.Session(config=tf.ConfigProto()) as sess:
        sess.run(tf.global_variables_initializer())
        saved_test_model = saver.save(sess, sess_test_filename, latest_filename='checkpoint_test')

    print('\n Test graph is generated and saved at: ' + saved_test_model)

    return True


def tf_complex2real(input_data):
    """
    Parameters
    ----------
    input_data : nrow x ncol.

    Returns
    -------
    outputs concatenated real and imaginary parts as nrow x ncol x 2

    """

    return tf.stack([tf.real(input_data), tf.imag(input_data)], axis=-1)


def tf_real2complex(input_data):
    """
    Parameters
    ----------
    input_data : nrow x ncol x 2

    Returns
    -------
    merges concatenated channels and outputs complex image of size nrow x ncol.

    """

    return tf.complex(input_data[..., 0], input_data[..., 1])


def tf_fftshift_flip2D(input_data, axes=1):
    """
    Parameters
    ----------
    input_data : ncoil x nrow x ncol
    axes :  The default is 1.
    ------

    """

    nx = math.ceil(args.nrow_GLOB / 2)
    ny = math.ceil(args.ncol_GLOB / 2)

    if axes == 1:

        first_half = tf.identity(input_data[:, :nx, :])
        second_half = tf.identity(input_data[:, nx:, :])

    elif axes == 2:

        first_half = tf.identity(input_data[:, :, :ny])
        second_half = tf.identity(input_data[:, :, ny:])

    else:
        raise ValueError('Invalid axes for fftshift')

    return tf.concat([second_half, first_half], axis=axes)


def tf_ifftshift_flip2D(input_data, axes=1):
    """
    Parameters
    ----------
    input_data : ncoil x nrow x ncol
    axes :  The default is 1.
    ------

    """

    nx = math.floor(args.nrow_GLOB / 2)
    ny = math.floor(args.ncol_GLOB / 2)

    if axes == 1:

        first_half = tf.identity(input_data[:, :nx, :])
        second_half = tf.identity(input_data[:, nx:, :])

    elif axes == 2:

        first_half = tf.identity(input_data[:, :, :ny])
        second_half = tf.identity(input_data[:, :, ny:])

    else:
        raise ValueError('Invalid axes for ifftshift')

    return tf.concat([second_half, first_half], axis=axes)


def tf_fftshift(input_x, axes=1):
    """
    Parameters
    ----------
    input_x : ncoil x nrow x ncol
    axes : The default is 1.

    """

    return tf_fftshift_flip2D(tf_fftshift_flip2D(input_x, axes=1), axes=2)


def tf_ifftshift(input_x, axes=1):
    """
    Parameters
    ----------
    input_x : ncoil x nrow x ncol
    axes : The default is 1.

    """

    return tf_ifftshift_flip2D(tf_ifftshift_flip2D(input_x, axes=1), axes=2)
