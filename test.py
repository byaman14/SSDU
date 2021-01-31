import os
import numpy as np
import tensorflow as tf
import scipy.io as sio
import matplotlib.pyplot as plt
import h5py as h5
import time
import utils
import parser_ops

parser = parser_ops.get_parser()
args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"

# .......................Load the Data...........................................
print('\n Loading ' + args.data_opt + ' test dataset...')
kspace_dir, coil_dir, mask_dir, saved_model_dir = utils.get_test_directory(args)

# %% kspace and sensitivity maps are assumed to be in .h5 format and mask is assumed to be in .mat
# Users can change these formats based on their dataset
kspace_test = h5.File(kspace_dir, "r")['kspace'][:]
sens_maps_testAll = h5.File(coil_dir, "r")['sens_maps'][:]
original_mask = sio.loadmat(mask_dir)['mask']

print('\n Normalize kspace to 0-1 region')
for ii in range(np.shape(kspace_test)[0]):
    kspace_test[ii, :, :, :] = kspace_test[ii, :, :, :] / np.max(np.abs(kspace_test[ii, :, :, :][:]))

# %% Train and loss masks are kept same as original mask during inference
nSlices, *_ = kspace_test.shape
test_mask = np.complex64(np.tile(original_mask[np.newaxis, :, :], (nSlices, 1, 1)))

print('\n size of kspace: ', kspace_test.shape, ', maps: ', sens_maps_testAll.shape, ', mask: ', test_mask.shape)

# %%  zeropadded outer edges of k-space with no signal- check github readme file for explanation for further explanations
# for coronal PD dataset, first 17 and last 16 columns of k-space has no signal
# in the training mask we set corresponding columns as 1 to ensure data consistency
if args.data_opt == 'Coronal_PD':
    test_mask[:, :, 0:17] = np.ones((nSlices, args.nrow_GLOB, 17))
    test_mask[:, :, 352:args.ncol_GLOB] = np.ones((nSlices, args.nrow_GLOB, 16))

test_refAll = np.empty((nSlices, args.nrow_GLOB, args.ncol_GLOB), dtype=np.complex64)
test_inputAll = np.empty((nSlices, args.nrow_GLOB, args.ncol_GLOB), dtype=np.complex64)

print('\n generating the refs and sense1 input images')
for ii in range(nSlices):
    sub_kspace = kspace_test[ii] * np.tile(test_mask[ii][..., np.newaxis], (1, 1, args.ncoil_GLOB))
    test_refAll[ii] = utils.sense1(kspace_test[ii, ...], sens_maps_testAll[ii, ...])
    test_inputAll[ii] = utils.sense1(sub_kspace, sens_maps_testAll[ii, ...])

sens_maps_testAll = np.transpose(sens_maps_testAll, (0, 3, 1, 2))
all_ref_slices, all_input_slices, all_recon_slices = [], [], []

print('\n  loading the saved model ...')
tf.reset_default_graph()
loadChkPoint = tf.train.latest_checkpoint(saved_model_dir)
config = tf.ConfigProto()
config.gpu_options.allow_growth = True

with tf.Session(config=config) as sess:
    new_saver = tf.train.import_meta_graph(saved_model_dir + '/model_test.meta')
    new_saver.restore(sess, loadChkPoint)

    # ..................................................................................................................
    graph = tf.get_default_graph()
    nw_output = graph.get_tensor_by_name('nw_output:0')
    nw_kspace_output = graph.get_tensor_by_name('nw_kspace_output:0')
    mu_param = graph.get_tensor_by_name('mu:0')
    x0_output = graph.get_tensor_by_name('x0:0')
    all_intermediate_outputs = graph.get_tensor_by_name('all_intermediate_outputs:0')

    # ...................................................................................................................
    trn_maskP = graph.get_tensor_by_name('trn_mask:0')
    loss_maskP = graph.get_tensor_by_name('loss_mask:0')
    nw_inputP = graph.get_tensor_by_name('nw_input:0')
    sens_mapsP = graph.get_tensor_by_name('sens_maps:0')
    weights = sess.run(tf.global_variables())

    for ii in range(nSlices):

        ref_image_test = np.copy(test_refAll[ii, :, :])[np.newaxis]
        nw_input_test = np.copy(test_inputAll[ii, :, :])[np.newaxis]
        sens_maps_test = np.copy(sens_maps_testAll[ii, :, :, :])[np.newaxis]
        testMask = np.copy(test_mask[ii, :, :])[np.newaxis]
        ref_image_test, nw_input_test = utils.complex2real(ref_image_test), utils.complex2real(nw_input_test)

        tic = time.time()
        dataDict = {nw_inputP: nw_input_test, trn_maskP: testMask, loss_maskP: testMask, sens_mapsP: sens_maps_test}
        nw_output_ssdu, *_ = sess.run([nw_output, nw_kspace_output, x0_output, all_intermediate_outputs, mu_param], feed_dict=dataDict)
        toc = time.time() - tic
        ref_image_test = utils.real2complex(ref_image_test.squeeze())
        nw_input_test = utils.real2complex(nw_input_test.squeeze())
        nw_output_ssdu = utils.real2complex(nw_output_ssdu.squeeze())

        if args.data_opt == 'Coronal_PD':
            """window levelling in presence of fully-sampled data"""
            factor = np.max(np.abs(ref_image_test[:]))
        else:
            factor = 1

        ref_image_test = np.abs(ref_image_test) / factor
        nw_input_test = np.abs(nw_input_test) / factor
        nw_output_ssdu = np.abs(nw_output_ssdu) / factor

        # ...............................................................................................................
        all_recon_slices.append(nw_output_ssdu)
        all_ref_slices.append(ref_image_test)
        all_input_slices.append(nw_input_test)

        print('\n Iteration: ', ii, 'elapsed time %f seconds' % toc)

plt.figure()
slice_num = 5
plt.subplot(1, 3, 1), plt.imshow(np.abs(all_ref_slices[slice_num]), cmap='gray'), plt.title('ref')
plt.subplot(1, 3, 2), plt.imshow(np.abs(all_input_slices[slice_num]), cmap='gray'), plt.title('input')
plt.subplot(1, 3, 3), plt.imshow(np.abs(all_recon_slices[slice_num]), cmap='gray'), plt.title('recon')
plt.show()
