import tensorflow as tf

import numpy as np


class estimator_1st_order(object):
    """
    This class implements a first order estimator of underlying parameter given an extensive set of evaluations
    """

    def __init__(self, sims, fiducial_point, offsets, print_params=False, tf_dtype=tf.float32,
                 tikohnov=0.0):
        """
        Initsalizes the first order estimator from a given evaluation of a function around a fiducial point
        :param sims: prediction of the fiducial point and its perturbations, shape [2*n_params + 1, n_sims, n_output]
        :param fiducial_point: the fiducial point of the expansion
        :param offsets: the perturbations used
        :param print_params: whether to print out the relevant parameters or not
        :param tf_dtype: the tensorflow dtype (float32 or float64)
        :param tikohnov: Add tikohnov regularization before inverting the jacobian
        """

        self.tf_dtype = tf_dtype

        # dimension check
        fidu_param = np.atleast_2d(fiducial_point)
        n_param = fidu_param.shape[-1]

        # get the fidu mean and cov
        sims = sims.astype(np.float64)
        fidu_sim = sims[0]

        # set fidu sims
        self.fidu_sim = fidu_sim.copy()

        fidu_mean = np.mean(fidu_sim, axis=0)
        fidu_cov = np.cov(fidu_sim, rowvar=False)

        # repeat the beginning
        fidu_sim = sims[0]
        fidu_mean = np.mean(fidu_sim, axis=0)
        fidu_cov = np.cov(fidu_sim, rowvar=False)

        # First we calculate the first order derivatives
        mean_derivatives = []
        cov_derivatives = []

        # to save the means
        means = []
        covs = []
        for i in range(n_param):
            # sims
            sims_minus = sims[2 * (i + 1) - 1]
            sims_plus = sims[2 * (i + 1)]

            # means
            mean_plus = np.mean(sims_plus, axis=0)
            mean_minus = np.mean(sims_minus, axis=0)

            # covariance
            cov_plus = np.cov(sims_plus, rowvar=False)
            cov_minus = np.cov(sims_minus, rowvar=False)

            # save
            means.append([mean_plus, mean_minus])
            covs.append([cov_plus, cov_minus])

            mean_derivatives.append((mean_plus - mean_minus) / (2.0 * offsets[i]))
            cov_derivatives.append((cov_plus - cov_minus) / (2.0 * offsets[i]))

        mean_jacobian = np.stack(mean_derivatives, axis=-1)
        cov_jacobian = np.stack(cov_derivatives, axis=-1)


        # calculate approximate fisher information
        # F = inv(J^-1 cov J^T^-1) = J^T cov^-1 J
        try:
            inv_cov = np.linalg.inv(fidu_cov)
        except:
            print("Covariance appears to be singular, using pseudo inverse...")
            inv_cov = np.linalg.pinv(fidu_cov)

        fisher = np.einsum('ij,jk->ik', inv_cov, mean_jacobian)
        fisher = np.einsum('ji,jk->ik', mean_jacobian, fisher)

        self.fisher = fisher

        # add regularization
        mean_jacobian += tikohnov*np.eye(mean_jacobian.shape[0], mean_jacobian.shape[1])

        # create a first order correction (we have pinv here as jac does not have to be square...)
        if mean_jacobian.shape[0] == mean_jacobian.shape[1]:
            inv_jac = np.linalg.inv(mean_jacobian)
        else:
            inv_jac = np.linalg.pinv(mean_jacobian)

        # set the other params
        self.mean_fidu = np.atleast_2d(fidu_mean)
        self.fidu_point = fidu_param
        self.inv_cov= inv_cov
        self.inv_jac = inv_jac

        self.fidu_point_tf = tf.constant(self.fidu_point, dtype=self.tf_dtype)
        self.inv_jac_tf = tf.constant(self.inv_jac, dtype=self.tf_dtype)
        self.mean_fidu_tf = tf.constant(self.mean_fidu, dtype=self.tf_dtype)

        # some info
        if print_params:
            print("Creating fisrt order estimator: fidu + J^-1(x-mu)")
            print("fidu: {}".format(self.fidu_point))
            print("J:    {}".format(mean_jacobian))
            print("J^-1: {}".format(self.inv_jac))
            print("mu:   {}".format(self.mean_fidu))
            print("\n Fiducial covariance: {}".format(fidu_cov))
            print("\n Derivative covariance: {}".format(cov_jacobian))

    def __call__(self, predictions, numpy=False):
        """
        Given some prediction it estimates to underlying parameters to first order
        :param predictions: The predictions i.e. summaries [n_summaries, n_output]
        :param numpy: perform the calculation in numpy instead of tensorflow
        :return: the estimates [n_summaries, n_output]
        """

        if numpy:
            return self.fidu_point + np.einsum("ij,aj->ai", self.inv_jac, predictions - self.mean_fidu)
        else:
            return self.fidu_point_tf + tf.einsum("ij,aj->ai", self.inv_jac_tf, predictions - self.mean_fidu_tf)
