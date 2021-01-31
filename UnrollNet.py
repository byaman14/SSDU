import tensorflow as tf
import data_consistency as ssdu_dc
import tf_utils
import models.networks as networks
import parser_ops

parser = parser_ops.get_parser()
args = parser.parse_args()


class UnrolledNet():
    """

    Parameters
    ----------
    input_x: batch_size x nrow x ncol x 2
    sens_maps: batch_size x ncoil x nrow x ncol

    trn_mask: batch_size x nrow x ncol, used in data consistency units
    loss_mask: batch_size x nrow x ncol, used to define loss in k-space

    args.nb_unroll_blocks: number of unrolled blocks
    args.nb_res_blocks: number of residual blocks in ResNet

    Returns
    ----------

    x: nw output image
    nw_kspace_output: k-space corresponding nw output at loss mask locations

    x0 : dc output without any regularization.
    all_intermediate_results: all intermediate outputs of regularizer and dc units
    mu: learned penalty parameter


    """

    def __init__(self, input_x, sens_maps, trn_mask, loss_mask):
        self.input_x = input_x
        self.sens_maps = sens_maps
        self.trn_mask = trn_mask
        self.loss_mask = loss_mask
        self.model = self.Unrolled_SSDU()

    def Unrolled_SSDU(self):
        x, denoiser_output, dc_output = self.input_x, self.input_x, self.input_x
        all_intermediate_results = [[0 for _ in range(2)] for _ in range(args.nb_unroll_blocks)]

        mu_init = tf.constant(0., dtype=tf.float32)
        x0 = ssdu_dc.dc_block(self.input_x, self.sens_maps, self.trn_mask, mu_init)

        with tf.name_scope('SSDUModel'):
            with tf.variable_scope('Weights', reuse=tf.AUTO_REUSE):
                for i in range(args.nb_unroll_blocks):
                    x = networks.ResNet(x, args.nb_res_blocks)
                    denoiser_output = x

                    mu = networks.mu_param()
                    rhs = self.input_x + mu * x

                    x = ssdu_dc.dc_block(rhs, self.sens_maps, self.trn_mask, mu)
                    dc_output = x

                    # ...................................................................................................
                    all_intermediate_results[i][0] = tf_utils.tf_real2complex(tf.squeeze(denoiser_output))
                    all_intermediate_results[i][1] = tf_utils.tf_real2complex(tf.squeeze(dc_output))

            nw_kspace_output = ssdu_dc.SSDU_kspace_transform(x, self.sens_maps, self.loss_mask)

        return x, nw_kspace_output, x0, all_intermediate_results, mu
