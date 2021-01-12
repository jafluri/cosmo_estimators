import numpy as np
from . import data

def log_likelihood(cl_obs, cosmo, wigner_mat, cov_mat, n_cov, n_of_zs, l_max=1000, bin_size=20):
    """
    Calculates the (unnormalized) log-likelihood given a set observation as described in the paper.
    """
    # check if l_max and bin_size are consistent
    if l_max%bin_size != 0:
        raise ValueError(f"Incompatible choice of l_max {l_max} and bin_size {bin_size}! The bin_size "
                         f"has to divide the l_max without remainder.")

    # We start by getting the spectra
    specs = data.gen_ccl_spectra(cosmo, n_of_zs, l_max=l_max+1)

    # we apply the wigner matrix and bin, note that l=0 is always ignored
    specs = [np.mean((wigner_mat.dot(cl[:l_max+1]))[1:].reshape((20, -1)), axis=1) for cl in specs]

    # difference to obs
    diff = np.concatenate(specs) - cl_obs

    # Get xS^-1x
    q = diff.dot(np.linalg.inv(cov_mat).dot(diff))

    # log prob
    loglikelihood = -0.5*n_cov*np.log(1.0 + q/(n_cov - 1))

    return loglikelihood