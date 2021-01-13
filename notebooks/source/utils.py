import numpy as np
import mpmath as mp

from . import kernels

def estimate_ABC_likelihood(predictions, fisher_matrix, observation, kernel="sigmoid", scale=0.5):
    """
    Estimates the ABC loglikelihood and its uncertainty from parameter estimations
    :param predictions: 2D array of predictions with shape [n_theta, n_preds, n_params]
    :param fisher_matrix: The (estimated) fisher matrix of the fiducial parameter with shape [n_params, n_params]
    :param observation: The estimator of the observation with shape [1, n_params]
    :param kernel: The kernel to use for the ABC estimation, either sigmoid, gauss or logistic
    :param scale: The scale used for the kernel.
    """
    # check kernel consistency
    if kernel == "sigmoid":
        kernel = lambda d: kernels.sigmoid_kernel(d, scale=scale, use_mp=True, return_mp=True)
    elif kernel == "gauss":
        kernel = lambda d: kernels.gaussian_kernel(d, scale=scale, use_mp=True, return_mp=True)
    elif kernel == "logistic":
        kernel = lambda d: kernels.logistic_kernel(d, scale=scale, use_mp=True, return_mp=True)
    else:
        raise ValueError(f"kernel type not understood {kernel}! kernel has to be either sigmoid, gauss or logistic.")

    # get the likelihood estimates
    loglike_estimates = []
    variance_estimates = []
    with mp.workdps(100):
        for i, pred in enumerate(predictions):
            d = np.einsum("ij,aj->ai", fisher_matrix, pred - observation)
            d = np.sqrt(np.sum(d * (pred - observation), axis=1))
            # get the kernal vals
            kernel_vals = kernel(d)
            # mean
            mean = kernels.mp_mean(kernel_vals)
            # variance of the mean
            var = kernels.mp_std(kernel_vals) ** 2 / len(d)
            log_like = np.float(mp.log(mean))
            # we add an eps here
            log_var = np.float(var / (mean ** 2 + 1e-99))
            loglike_estimates.append(log_like)
            variance_estimates.append(log_var)

    # concat to initial value
    Y_init = np.concatenate([np.asarray(loglike_estimates).reshape((-1, 1)),
                             np.asarray(variance_estimates).reshape((-1, 1))], axis=1)
    return Y_init

def ij_to_list_index(i, j, n):
    """
    Assuming you have a symetric nxn matrix M and take the entries of the upper triangular including the
    diagonal and then ravel it to transform it into a list. This function will transform a matrix location
    given row i and column j into the proper list index.
    :param i: row index of the matrix M
    :param j: column index of matrix M
    :param n: total number of rows / colums of n
    :return: The correct index of the lost
    """
    assert j >= i, "We only consider the upper triangular part..."

    index = 0
    for k in range(i):
        index += n - k - 1
    return index + j