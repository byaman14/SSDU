import numpy as np
from skimage.measure import compare_ssim


def get_train_directory(args):
    """
    Parameters
    ----------
    args :  args.data_opt--dataset to be used in training&testing
    Note: users should set the directories prior to running train file
    Returns
    -------
    directories of the kspace, sensitivity maps and mask

    """

    if args.data_opt == 'Coronal_PD':

        kspace_dir = '...'
        coil_dir = '...'

    elif args.data_opt == 'Coronal_PDFS':

        kspace_dir = '...'
        coil_dir = '...'

    else:
        raise ValueError('Invalid data option')

    mask_dir = '...'

    print('\n kspace dir : ', kspace_dir, '\n \n coil dir :', coil_dir, '\n \n mask dir: ', mask_dir)

    return kspace_dir, coil_dir, mask_dir


def get_test_directory(args):
    """
    Parameters
    ----------
    args :  args.data_opt--dataset to be used in training&testing
    Note: users should set the directories prior to running test file
    Returns
    -------
    directories of the kspace, sensitivity maps and mask and ;
    saved_model_dir : saved training model directory

    """
    if args.data_opt == 'Coronal_PD':

        kspace_dir = '...'
        coil_dir = '...'
        saved_model_dir = '../SSDU_Coronal_PD_100Epochs_Rate4_10Unrolls_GaussianSelection'

    elif args.data_opt == 'Coronal_PDFS':

        kspace_dir = '...'
        coil_dir = '...'
        saved_model_dir = '...'

    else:
        raise ValueError('Invalid data option')

    mask_dir = '...'

    print('\n kspace dir : ', kspace_dir, '\n \n coil dir :', coil_dir,
          '\n \n mask dir: ', mask_dir, '\n \n saved model dir: ', saved_model_dir)

    return kspace_dir, coil_dir, mask_dir, saved_model_dir


def getSSIM(space_ref, space_rec):
    """
    Measures SSIM between the reference and the reconstructed images
    """

    space_ref = np.squeeze(space_ref)
    space_rec = np.squeeze(space_rec)
    space_ref = space_ref / np.amax(np.abs(space_ref))
    space_rec = space_rec / np.amax(np.abs(space_ref))
    data_range = np.amax(np.abs(space_ref)) - np.amin(np.abs(space_ref))

    return compare_ssim(space_rec, space_ref, data_range=data_range,
                        gaussian_weights=True,
                        use_sample_covariance=False)


def getPSNR(ref, recon):
    """
    Measures PSNR between the reference and the reconstructed images
    """

    mse = np.sum(np.square(np.abs(ref - recon))) / ref.size
    psnr = 20 * np.log10(np.abs(ref.max()) / (np.sqrt(mse) + 1e-10))

    return psnr


def fft(ispace, axes=(0, 1), norm=None, unitary_opt=True):
    """
    Parameters
    ----------
    ispace : coil images of size nrow x ncol x ncoil.
    axes :   The default is (0, 1).
    norm :   The default is None.
    unitary_opt : The default is True.

    Returns
    -------
    transform image space to k-space.

    """

    kspace = np.fft.fftshift(np.fft.fftn(np.fft.ifftshift(ispace, axes=axes), axes=axes, norm=norm), axes=axes)

    if unitary_opt:

        fact = 1

        for axis in axes:
            fact = fact * kspace.shape[axis]

        kspace = kspace / np.sqrt(fact)

    return kspace


def ifft(kspace, axes=(0, 1), norm=None, unitary_opt=True):
    """
    Parameters
    ----------
    ispace : image space of size nrow x ncol x ncoil.
    axes :   The default is (0, 1).
    norm :   The default is None.
    unitary_opt : The default is True.

    Returns
    -------
    transform k-space to image space.

    """

    ispace = np.fft.ifftshift(np.fft.ifftn(np.fft.fftshift(kspace, axes=axes), axes=axes, norm=norm), axes=axes)

    if unitary_opt:

        fact = 1

        for axis in axes:
            fact = fact * ispace.shape[axis]

        ispace = ispace * np.sqrt(fact)

    return ispace


def norm(tensor, axes=(0, 1, 2), keepdims=True):
    """
    Parameters
    ----------
    tensor : It can be in image space or k-space.
    axes :  The default is (0, 1, 2).
    keepdims : The default is True.

    Returns
    -------
    tensor : applies l2-norm .

    """
    for axis in axes:
        tensor = np.linalg.norm(tensor, axis=axis, keepdims=True)

    if not keepdims: return tensor.squeeze()

    return tensor


def find_center_ind(kspace, axes=(1, 2, 3)):
    """
    Parameters
    ----------
    kspace : nrow x ncol x ncoil.
    axes :  The default is (1, 2, 3).

    Returns
    -------
    the center of the k-space

    """

    center_locs = norm(kspace, axes=axes).squeeze()

    return np.argsort(center_locs)[-1:]


def index_flatten2nd(ind, shape):
    """
    Parameters
    ----------
    ind : 1D vector containing chosen locations.
    shape : shape of the matrix/tensor for mapping ind.

    Returns
    -------
    list of >=2D indices containing non-zero locations

    """

    array = np.zeros(np.prod(shape))
    array[ind] = 1
    ind_nd = np.nonzero(np.reshape(array, shape))

    return [list(ind_nd_ii) for ind_nd_ii in ind_nd]


def sense1(input_kspace, sens_maps, axes=(0, 1)):
    """
    Parameters
    ----------
    input_kspace : nrow x ncol x ncoil
    sens_maps : nrow x ncol x ncoil

    axes : The default is (0,1).

    Returns
    -------
    sense1 image

    """

    image_space = ifft(input_kspace, axes=axes, norm=None, unitary_opt=True)
    Eh_op = np.conj(sens_maps) * image_space
    sense1_image = np.sum(Eh_op, axis=axes[-1] + 1)

    return sense1_image


def complex2real(input_data):
    """
    Parameters
    ----------
    input_data : row x col
    dtype :The default is np.float32.

    Returns
    -------
    output : row x col x 2

    """

    return np.stack((input_data.real, input_data.imag), axis=-1)


def real2complex(input_data):
    """
    Parameters
    ----------
    input_data : row x col x 2

    Returns
    -------
    output : row x col

    """

    return input_data[..., 0] + 1j * input_data[..., 1]
