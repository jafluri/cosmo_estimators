import horovod.tensorflow as hvd
import tensorflow as tf

import numpy as np



def _get_backend_floatx():
    """
    returns the current backend float of the keras backend
    :return: either tf.float32 or tf.float64 depending on the current backend setting
    """
    if tf.keras.backend.floatx() == 'float32':
        return tf.float32
    elif tf.keras.backend.floatx() == 'float64':
        return tf.float64
    else:
        raise ValueError(f"The only suppored keras backend floatx are float64 and float32 not "
                         f"{tf.keras.backend.floatx()}! Please use tf.keras.backend.set_floatx to set an appropiate "
                         f"value.")

def tf_matrix_condition(m):
    """
    Calculated the matrix condition number of an input m over the last two axis, defined as the ratio of the largest
    and smallest singular value
    :param m: The input tensorf [...,N,M]
    :return: The condition number [...]
    """
    s = tf.linalg.svd(m, compute_uv=False)
    return s[..., 0] / s[..., -1]

def get_jac_and_cov_matrix(predictions, n_params, n_same, off_sets, n_output=None, summary_writer=None, training=False,
                           num_workers=None):
    """
    Calculates the covariance of the fiducial predictions and the jacobians of the means and returns it.
    It assumes a specific ordering of the predictions.
    :param predictions: predictions
    :param n_params: number of underlying model parameters
    :param n_same: how many of the same parameter
    :param off_sets: the difference in underlying parameter to calculate the jacobian
    :param n_output: dimensionality of the summary statistic, defaults to predictions.get_shape()[-1]
    :param summary_writer: summary writer used to write tensorboard summaries
    :param training: if we are currently training, if False, no summary is written even if a writer is provided
    :param num_workers: number of workers in total, None means no parallel run
    :return: covariance and jacobians
    """
    # get the current backend
    current_float = _get_backend_floatx()

    # in case predctions is a numpy array
    if isinstance(predictions, np.ndarray):
        predictions = tf.convert_to_tensor(predictions, dtype=current_float)

    # get number of outputs
    if n_output is None:
        n_output = int(predictions.get_shape()[-1])

    # split the output
    splits = [tf.reshape(split, shape=[-1, n_same, n_output])
              for split in tf.split(predictions, num_or_size_splits=2 * n_params + 1, axis=0)]

    # collect if necessary
    if num_workers is not None:
        splits = [tf.transpose(hvd.allgather(tf.transpose(split, perm=[1,0,2])), perm=[1,0,2]) for split in splits]
        n_cov = num_workers*n_same - 1.0
    else:
        n_cov = n_same - 1.0

    # summary
    if training:
        param_splits = tf.split(splits[0], num_or_size_splits=n_output, axis=-1)
        for num, single_param in enumerate(param_splits):
            if summary_writer is not None:
                with summary_writer.as_default():
                    tf.summary.histogram("Param_%i_hist" % (num), single_param)

    # get the covariance
    mean = tf.reduce_mean(splits[0], axis=1, keepdims=True)
    outmm = tf.subtract(splits[0], mean)
    cov = tf.divide(tf.einsum('hjk,hjl->hkl', outmm, outmm),
                    n_same - 1.0, name="COV")

    # Handle off sets an renormalization
    off_sets = tf.convert_to_tensor(off_sets, dtype=current_float)

    # get mean derivatives
    derivatives = []
    for i in range(n_params):
        mean_minus = tf.reduce_mean(splits[2 * (i + 1) - 1], axis=1, keepdims=False)
        mean_plus = tf.reduce_mean(splits[2 * (i + 1)], axis=1, keepdims=False)
        derivatives.append(tf.divide(tf.subtract(mean_plus, mean_minus),
                                     tf.scalar_mul(2.0, off_sets[i])))

    # stack the derivatives to form jacobian
    jacobian = tf.stack(derivatives, axis=-1)

    return cov, jacobian

def get_fisher_from_cov_jacobian(cov, jacobian):
    """
    Calculates the approximate fisher information given a covariance matrix and jacobian
    :param cov: the covariance matrix of the summary
    :param jacobian: the jacobian of the summary
    :return: the approximate fisher matrix
    """

    # calculate approximate fisher information
    # F = inv(J^-1 cov J^T^-1) = J^T cov^-1 J
    inv_cov = tf.linalg.inv(cov)
    fisher = tf.einsum('aij,ajk->aik', inv_cov, jacobian)
    fisher = tf.einsum('aji,ajk->aik', jacobian, fisher)

    return fisher

def delta_loss(predictions, n_params, n_same, off_sets, n_output=None, jac_weight=0.0, force_params=None,
               force_weight=1.0, jac_cond_weight=None, use_log_det=True, no_correlations=False, tikhonov_regu=True,
               summary_writer=None, training=True, weights=None, eps=1e-32, n_partial=None, num_workers=None,
               img_summary=False):
    """
    This function calculates the delta loss which tries to maximize the information of the summary statistics. Note it
    needs the predtions to be ordered in a specific way:
        * The shape of the predictions is (n_points*n_same*(2*n_params+1), n_output)
        * If one splits the predictions into (2*n_params+1) parts among the first axis one has the following scheme:
            * The first part was generated with the unperturbed parameters
            * The second part was generated with parameters where off_sets[0] was subtracted from the first param
            * The third part was generated with parameters where off_sets[0] was added from to first param
            * The fourth part was generated with parameters where off_sets[1] was subtracted from the second param
            * and so on
    :param predictions: The predictions a.k.a. summary statistics in a specific ordering
    :param n_params: Number of underlying model parameter
    :param n_same: How many summaries (unperturbed) are coming from the same parameter set
    :param off_sets: The off_sets used to perturb the original parameters and used for the Jacobian calculation
    :param n_output: Dimensionality of the summary statistic, defaults to predictions.get_shape()[-1]
    :param jac_weight: The weight of the Jacobian loss (loss that forces the Jacobian of the summaries to be close to
                       unity (or identity matrix).
    :param force_params: Either None or a set of parameters with shape (n_points, 1, n_output) which is used to compute
                         a square loss of the unperturbed summaries. It is useful to set this for example to zeros
                         such that the network does not produces arbitrary high summary values
    :param force_weight: The weight of the square loss of force_params
    :param jac_cond_weight: If not None, this weight is used to add an additional loss using the matrix condition
                            number of the jacobian
    :param use_log_det: Use the log of the determinants in the information inequality, should be True. If False the
                        information inequality is not minimized in a proper manner and the training can become unstable.
    :param no_correlations: Do not consider correlations between the parameter, this means that one tries to find
                                an optimal summary (single value) for each underlying model parameter, only possible
                                if n_output == n_params
    :param tikhonov_regu: Use Tikhonov regularization of matrices e.g. to avoid vanishing determinants. This is the
                          recommended regularization method as it allows the usage of some optimized routines.
    :param summary_writer: The writer used to write tensorboard summaries.
    :param training: Whether the loss is used for training. If False, no summaries will be written even if a
                     summary_writer is supplied.
    :param weights: An 1d array of length n_points, used as weights in means of the different points.
    :param eps: A small positive value used for regularization of things like logs etc. This should only be increased
                if tikhonov_regu is used and a error is raised.
    :param n_partial: To train only on a subset of parameters and not all underlying model parameter. Defaults to None
                      which means the information inequality is minimized in a normal fashion. Note that due to the
                      necessity of some algebraic manipulations n_partial == None and n_partial == n_params lead to
                      slightly different behaviour.
    :param num_workers: number of workers in total, None means no parallel run
    :param img_summary: image summary of jacobian and covariance
    :return: The loss value.
    """
    # TODO: A fixed epsilon can lead to some problems. E.g. in tikonov regularization might fail because the lack
    # TODO: of precision. A possible solution would be to use the machine epsilon for added regularization
    # TODO: and a fixed epsilon for absolut regulatization (division or log errors...)

    cov, jacobian = get_jac_and_cov_matrix(predictions=predictions, n_params=n_params, n_same=n_same, off_sets=off_sets,
                                           n_output=n_output, summary_writer=summary_writer, training=training)

    if training and summary_writer is not None:
        with summary_writer.as_default():
            tf.summary.histogram("Jacobian", jacobian)
            if img_summary:
                jac_img = tf.expand_dims(jacobian, axis=3)
                jac_max = tf.reduce_max(jac_img, axis=(1, 2), keepdims=True)
                jac_min = tf.reduce_min(jac_img, axis=(1, 2), keepdims=True)
                jac_img = tf.math.divide(jac_img - jac_min, jac_max - jac_min)
                jac_img = tf.image.resize(jac_img, size=(128, 128), method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
                tf.summary.image("JacobianIMG", jac_img)
                cov_img = tf.expand_dims(cov, axis=3)
                cov_max = tf.reduce_max(cov_img, axis=(1, 2), keepdims=True)
                cov_min = tf.reduce_min(cov_img, axis=(1, 2), keepdims=True)
                cov_img = tf.math.divide(cov_img - cov_min, cov_max - cov_min)
                cov_img = tf.image.resize(cov_img, size=(128, 128), method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
                tf.summary.image("CovarianceIMG", cov_img)
                # get corrlation matrix
                v = tf.math.sqrt(tf.linalg.diag_part(cov))
                outer_v = tf.einsum("ai,aj->aij",v,v)
                cor = tf.divide(cov, outer_v)
                cor_img = tf.expand_dims(cor, axis=3)
                # fit between 0 and 1
                cor_img = tf.math.add(0.5, tf.math.scalar_mul(0.5, cor_img))
                # to image with good resolution
                cor_img = tf.image.resize(cor_img, size=(128, 128), method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
                tf.summary.image("CorrelationIMG", cor_img)

    # get the current backend
    current_float = _get_backend_floatx()

    # in case predctions is a numpy array
    if isinstance(predictions, np.ndarray):
        predictions = tf.convert_to_tensor(predictions, dtype=current_float)

    # get number of outputs
    if n_output is None:
        n_output = int(predictions.get_shape()[-1])

    # note worthy stuff
    if n_output > n_params and jac_weight > 0.0:
        print("WARNING: The weight of the Jacobian loss should be zero, if you have a summary that has a higher "
              "dimension as the number of model params!", flush=True)
    if no_correlations and (n_output != n_params):
        raise ValueError("Independent summaries (no_correlations) is only possible if n_output == n_params")
    if no_correlations and n_partial is not None:
        raise ValueError("Independent summaries (no_correlations) is only possible if n_partial is None")

    if use_log_det:
        # check if we are in no correlation regime
        if no_correlations:
            # tikhonov_regu and normal regu is the same in this case
            cov_diag = tf.linalg.diag_part(cov)
            cov_log_det = tf.math.log(cov_diag + eps)
            jac_diag = tf.linalg.diag_part(jacobian)
            jac_log_det = tf.math.log(tf.square(jac_diag) + eps)
            # the factor of 2 is in the square of the jac_diag
            cov_det = tf.reduce_mean(tf.subtract(cov_log_det, jac_log_det))

        # use everything
        elif n_partial is None:
            # tf.logdet is much better for the backprob, but fails if the det is zero
            # should we do cov + eps*identity?
            if tikhonov_regu:
                identity = tf.scalar_mul(eps, tf.eye(n_params, batch_shape=[1]))
                cov_log_det = tf.linalg.logdet(tf.add(cov, identity))
                # we use that 2*log(det(A)) = log(det(A)^2) = log(det(A)*det(A)) = log(det(A)*det(A^T))
                #                           = log(det(A*A^T))
                with tf.name_scope("jac_logdet") as scope:
                   jt_j = tf.einsum("aji,ajk->aik", jacobian, jacobian)
                   jac_log_det = tf.linalg.logdet(tf.add(jt_j, identity))
                with tf.name_scope("cov_logdet") as scope:
                    cov_det = tf.subtract(cov_log_det, jac_log_det)
            else:
                # We add a abs here because of instabilities
                cov_log_det = tf.math.log(tf.math.abs(tf.linalg.det(cov)) + eps)
                with tf.name_scope("jac_logdet") as scope:
                    jac_log_det = tf.math.log(tf.math.abs(tf.linalg.det(jacobian)) + eps)
                with tf.name_scope("cov_logdet") as scope:
                    cov_det = tf.subtract(cov_log_det, tf.scalar_mul(2.0, jac_log_det))
        else:
            # we use only the first n_partial params
            j_part = jacobian[:,:,:n_partial]

            # now we need to calculate log(det(J^T cov J)) - log(det(J^T J))
            cov_j = tf.einsum("aij,ajk->aik", cov, j_part)
            jt_cov_j = tf.einsum("aji,ajk->aik", j_part, cov_j)
            jt_j = tf.einsum("aji,ajk->aik", j_part, j_part)

            if tikhonov_regu:
                id_dim = np.minimum(n_params, n_partial)
                identity = tf.scalar_mul(eps, tf.eye(id_dim, batch_shape=[1]))
                with tf.name_scope("jac_logdet") as scope:
                    jac_log_det = tf.linalg.logdet(tf.add(jt_j , identity))
                with tf.name_scope("cov_logdet") as scope:
                    cov_log_det = tf.linalg.logdet(tf.add(jt_cov_j, identity))
            else:
                # We add a abs here because of instabilities
                with tf.name_scope("jac_logdet") as scope:
                    jac_log_det = tf.math.log(tf.math.abs(tf.linalg.det(jt_j)) + eps)
                with tf.name_scope("cov_logdet") as scope:
                    cov_log_det = tf.math.log(tf.math.abs(tf.linalg.det(jt_cov_j)) + eps)

            cov_det = tf.subtract(cov_log_det, tf.scalar_mul(2.0, jac_log_det))
    else:
        # dividing by the jac_det (for info inequality) does not work...
        print(f"WARNING: You are using use_log_det=False. Only the determinant of the covariance matrix will be "
              f"optimized. This loss might be unbouned and could lead to unstable training.")
        cov_det = tf.linalg.det(cov)

    if weights is not None:
        # normalize the weights
        weights = tf.divide(weights, tf.reduce_sum(weights))
        # do a weighted mean
        cov_det = tf.multiply(weights, cov_det)
    # normal mean
    cov_det = tf.reduce_mean(cov_det)
    if training and summary_writer is not None:
        with summary_writer.as_default():
            tf.summary.scalar("Cov_loss_det", cov_det)

    # jacobian loss (log of this is unstable)
    diff = tf.subtract(jacobian, tf.expand_dims(tf.eye(n_output, n_params), axis=0))
    if n_partial is None:
        # use everything
        jac_loss = tf.reduce_mean(tf.square(diff), axis=(1,2))
    else:
        # only n_part
        jac_loss = tf.reduce_mean(tf.square(diff)[:,:,:n_partial], axis=(1, 2))


    if weights is not None:
        jac_loss = tf.multiply(weights, jac_loss)
    jac_loss = tf.reduce_mean(jac_loss)
    if training and summary_writer is not None:
        with summary_writer.as_default():
            tf.summary.scalar("Jacobian_loss", jac_loss)

    loss = tf.add(cov_det, tf.scalar_mul(jac_weight, jac_loss))

    # condition number loss
    if jac_cond_weight is not None:
        if n_partial is not None:
            c = tf_matrix_condition(jacobian[...,:n_partial])
        else:
            c = tf_matrix_condition(jacobian)
        if weights is not None:
            c = tf.multiply(weights, c)
        jac_cond_loss = tf.reduce_mean(c)
        if training and summary_writer is not None:
            with summary_writer.as_default():
                tf.summary.scalar("Jacobian_Condition", jac_cond_loss)
        jac_cond_loss = tf.scalar_mul(jac_cond_weight, jac_cond_loss)
        loss = tf.add(loss, jac_cond_loss)

    if force_params is not None:
        # calculate square distance between fidu mean and preds
        mid_params = tf.split(predictions, num_or_size_splits=2 * n_params + 1, axis=0)[0]

        # reshape
        mid_params = tf.reshape(mid_params, shape=[-1, n_same, n_output])

        # penalty
        diff = tf.subtract(mid_params, force_params)
        diff_loss = tf.square(tf.reduce_mean(diff, axis=1))

        if weights is not None:
            # reduce mean over the last axis (n params)
            diff_loss = tf.reduce_mean(diff_loss, axis=1)
            # weight and mean
            diff_loss = tf.multiply(diff_loss, weights)
        # simple mean reduction
        diff_loss = tf.reduce_mean(diff_loss)

        # summary
        if training and summary_writer is not None:
            # we need to mean this between worker if necessary
            if num_workers is not None:
                diff_loss_sum = hvd.allreduce(diff_loss)
            else:
                diff_loss_sum = diff_loss
            with summary_writer.as_default():
                tf.summary.scalar("Diff_loss", diff_loss_sum)

        # force weight
        diff_loss = tf.scalar_mul(force_weight, diff_loss)

        # add to loss
        loss = tf.add(loss, diff_loss)

    return loss
