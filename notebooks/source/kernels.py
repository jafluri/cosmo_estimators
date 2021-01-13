import mpmath as mp
import numpy as np

"""
This file contains function with the mpmath library featuring arbitrary precise floats
"""

# Note: for float32 this is pretty close to the min logable val
min_val = 1e-42

@mp.workdps(100)
def gaussian_kernel(d, scale=0.05, use_mp=False, return_mp=False):
    """
    Gaussian kernel with scale
    :param d: the input of the kernel, might be numpy array but not mp matrix
    :param scale: the scale
    :param use_mp: if True use mpmath
    :param return_mp: if True return mpf object
    :return: the kernel evaluated at d and scale
    """
    if not use_mp:
        if return_mp:
            print("To return an mpf object, set use_mp=True, returning numpy object...")
        norm = np.sqrt(2 * np.pi) * scale
        chi = -0.5 * (d / scale) ** 2
        return np.maximum(np.exp(chi) / (norm), min_val)

    else:
        if isinstance(d, np.ndarray):
            is_array = True
            shape = d.shape
            d = d.ravel().astype(np.float64)
        else:
            is_array = False
            d = np.array([d]).astype(np.float64)

        res = []
        for dd in d:
            norm = mp.sqrt(2 * mp.pi) * scale
            chi = -0.5 * (mp.mpf(dd) / scale) ** 2
            res.append(mp.exp(chi) / norm)

        if is_array:
            if return_mp:
                return np.array(res).reshape(shape)
            else:
                return np.array(res, dtype=np.float64).reshape(shape)
        else:
            if return_mp:
                return res[0]
            else:
                return np.float64(res[0])




@mp.workdps(100)
def logistic_kernel(d, scale=0.05, use_mp=False, return_mp=False):
    """
    logistic kernel with scale
    :param d: the input of the kernel, might be numpy array but not mp matrix
    :param scale: the scale
    :param use_mp: if True use mpmath
    :param return_mp: if True return mpf object
    :return: the kernel evaluated at d and scale
    """
    if not use_mp:
        if return_mp:
            print("To return an mpf object, set use_mp=True, returning numpy object...")
        pos_exp = np.exp(d / scale)
        neg_exp = np.exp(-d / scale)
        return np.maximum(1.0 / (scale * (pos_exp + neg_exp + 2)), min_val)

    else:
        if isinstance(d, np.ndarray):
            is_array = True
            shape = d.shape
            d = d.ravel().astype(np.float64)
        else:
            is_array = False
            d = np.array([d]).astype(np.float64)

        res = []
        for dd in d:
            pos_exp = mp.exp(mp.mpf(dd) / scale)
            neg_exp = mp.exp(-mp.mpf(dd) / scale)
            res.append(1.0 / (scale * (pos_exp + neg_exp + 2.0)))

        if is_array:
            if return_mp:
                return np.array(res).reshape(shape)
            else:
                return np.array(res, dtype=np.float64).reshape(shape)
        else:
            if return_mp:
                return res[0]
            else:
                return np.float64(res[0])


@mp.workdps(100)
def sigmoid_kernel(d, scale=0.05, use_mp=False, return_mp=False):
    """
    Sigmoid kernel with scale
    :param d: the input of the kernel, might be numpy array but not mp matrix
    :param scale: the scale
    :param use_mp: if True use mpmath
    :param return_mp: if True return mpf object
    :return: the kernel evaluated at d and scale
    """
    if not use_mp:
        if return_mp:
            print("To return an mpf object, set use_mp=True, returning numpy object...")
        pos_exp = np.exp(d/scale)
        neg_exp = np.exp(-d/scale)
        return np.maximum(2.0/(np.pi*scale*(pos_exp + neg_exp)), min_val)

    else:
        if isinstance(d, np.ndarray):
            is_array = True
            shape = d.shape
            d = d.ravel().astype(np.float64)
        else:
            is_array = False
            d = np.array([d]).astype(np.float64)

        res = []
        for dd in d:
            pos_exp = mp.exp(mp.mpf(dd) / scale)
            neg_exp = mp.exp(-mp.mpf(dd) / scale)
            res.append(2.0 / (mp.pi * scale * (pos_exp + neg_exp)))

        if is_array:
            if return_mp:
                return np.array(res).reshape(shape)
            else:
                return np.array(res, dtype=np.float64).reshape(shape)
        else:
            if return_mp:
                return res[0]
            else:
                return np.float64(res[0])


# mpmath functions
@mp.workdps(100)
def mp_mean(arr):
    """
    Calculates the mean of the array of mpf values
    :param arr: array of mp.mpf floats
    :return: the mean as mpf
    """
    arr = arr.ravel()
    N = arr.size

    res = mp.mpf(0.0)
    for a in arr:
        res = res + a

    return res/N


@mp.workdps(100)
def mp_std(arr, ddof=0):
    """
    Calculates the standard deviation of the array of mpf values
    :param arr: array of mp.mpf floats
    :param ddof: Means Delta Degrees of Freedom. The divisor used in calculations is N - ddof,
    where N represents the number of elements. By default ddof is zero. (np.std convention)
    :return: the mean as mpf
    """
    arr = arr.ravel()
    N = arr.size

    # get the mean
    mean = mp_mean(arr)

    res = mp.mpf(0.0)
    for a in arr:
        res = res + (a - mean)**2

    return mp.sqrt(res / (N - ddof))
