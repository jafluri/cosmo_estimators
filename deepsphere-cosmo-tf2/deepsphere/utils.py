"""Utilities module."""

import numpy as np
from scipy import sparse
import healpy as hp


def extend_indices(indices, nside_in, nside_out, nest=True):
    """
    Minimally extends a set of indices such that it can be reduced to nside_out in a healpy fashion, always four pixels
    reduce naturally to a higher order pixel. Note that this function supports the ring ordering, however, since almost
    no other function does so, nest ordering is strongly recommended.
    :param indices: 1d array of integer pixel ids
    :param nside_in: nside of the input
    :param nside_out: nside of the output
    :param nest: indices are ordered in the "NEST" ordering scheme
    :return: returns a set of indices in the same ordering as the input.
    """
    # figire out the ordering
    if nest:
        ordering = "NEST"
    else:
        ordering = "RING"

    # get the map to reduce
    m_in = np.zeros(hp.nside2npix(nside_in))
    m_in[indices] = 1.0

    # reduce
    m_in = hp.ud_grade(map_in=m_in, nside_out=nside_out, order_in=ordering, order_out=ordering)

    # expand
    m_in = hp.ud_grade(map_in=m_in, nside_out=nside_in, order_in=ordering, order_out=ordering)

    # get the new indices
    return np.arange(hp.nside2npix(nside_in))[m_in > 1e-12]


def healpix_weightmatrix(nside=16, nest=True, indexes=None, dtype=np.float32):
    """Return an unnormalized weight matrix for a graph using the HEALPIX sampling.

    Parameters
    ----------
    nside : int
        The healpix nside parameter, must be a power of 2, less than 2**30.
    nest : bool, optional
        if True, assume NESTED pixel ordering, otherwise, RING pixel ordering
    indexes : list of int, optional
        List of indexes to use. This allows to build the graph from a part of
        the sphere only. If None, the default, the whole sphere is used.
    dtype : data-type, optional
        The desired data type of the weight matrix.
    """
    if not nest:
        raise NotImplementedError()

    if indexes is None:
        indexes = range(nside**2 * 12)
    npix = len(indexes)  # Number of pixels.
    if npix >= (max(indexes) + 1):
        # If the user input is not consecutive nodes, we need to use a slower
        # method.
        usefast = True
        indexes = range(npix)
    else:
        usefast = False
        indexes = list(indexes)

    # Get the coordinates.
    x, y, z = hp.pix2vec(nside, indexes, nest=nest)
    coords = np.vstack([x, y, z]).transpose()
    coords = np.asarray(coords, dtype=dtype)
    # Get the 7-8 neighbors.
    neighbors = hp.pixelfunc.get_all_neighbours(nside, indexes, nest=nest)

    # Indices of non-zero values in the adjacency matrix.
    col_index = neighbors.T.reshape((npix * 8))
    row_index = np.repeat(indexes, 8)

    # Remove pixels that are out of our indexes of interest (part of sphere).
    if usefast:
        keep = (col_index < npix)
        # Remove fake neighbors (some pixels have less than 8).
        keep &= (col_index >= 0)
        col_index = col_index[keep]
        row_index = row_index[keep]
    else:
        col_index_set = set(indexes)
        keep = [c in col_index_set for c in col_index]
        inverse_map = [np.nan] * (nside**2 * 12)
        for i, index in enumerate(indexes):
            inverse_map[index] = i
        col_index = [inverse_map[el] for el in col_index[keep]]
        row_index = [inverse_map[el] for el in row_index[keep]]

    # Compute Euclidean distances between neighbors.
    distances = np.sum((coords[row_index] - coords[col_index])**2, axis=1)
    # slower: np.linalg.norm(coords[row_index] - coords[col_index], axis=1)**2

    # Compute similarities / edge weights.
    kernel_width = np.mean(distances)
    weights = np.exp(-distances / (2 * kernel_width))

    # Similarity proposed by Renata & Pascal, ICCV 2017.
    # weights = 1 / distances

    # Build the sparse matrix.
    W = sparse.csr_matrix(
        (weights, (row_index, col_index)), shape=(npix, npix), dtype=dtype)

    return W


def build_laplacian(W, lap_type='normalized', dtype=np.float32):
    """Build a Laplacian (tensorflow)."""
    d = np.ravel(W.sum(1))
    if lap_type == 'combinatorial':
        D = sparse.diags(d, 0, dtype=dtype)
        return (D - W).tocsc()
    elif lap_type == 'normalized':
        d12 = np.power(d, -0.5)
        D12 = sparse.diags(np.ravel(d12), 0, dtype=dtype).tocsc()
        return sparse.identity(d.shape[0], dtype=dtype) - D12 * W * D12
    else:
        raise ValueError('Unknown Laplacian type {}'.format(lap_type))


def healpix_graph(nside=16,
                  nest=True,
                  lap_type='normalized',
                  indexes=None,
                  use_4=False,
                  dtype=np.float32):
    """Build a healpix graph using the pygsp from NSIDE."""
    from pygsp import graphs

    if indexes is None:
        indexes = range(nside**2 * 12)

    # 1) get the coordinates
    npix = hp.nside2npix(nside)  # number of pixels: 12 * nside**2
    pix = range(npix)
    x, y, z = hp.pix2vec(nside, pix, nest=nest)
    coords = np.vstack([x, y, z]).transpose()[indexes]
    # 2) computing the weight matrix
    if use_4:
        raise NotImplementedError()
        # W = build_matrix_4_neighboors(nside, indexes, nest=nest, dtype=dtype)
    else:
        W = healpix_weightmatrix(
            nside=nside, nest=nest, indexes=indexes, dtype=dtype)
    # 3) building the graph
    G = graphs.Graph(
        W,
        lap_type=lap_type,
        coords=coords)
    return G


def healpix_laplacian(nside=16,
                      nest=True,
                      lap_type='normalized',
                      indexes=None,
                      dtype=np.float32,
                      use_4=False):
    """Build a Healpix Laplacian."""
    if use_4:
        raise NotImplementedError("use_4 is not implemented!")
    else:
        W = healpix_weightmatrix(
            nside=nside, nest=nest, indexes=indexes, dtype=dtype)
    L = build_laplacian(W, lap_type=lap_type)
    return L


def rescale_L(L, lmax=2, scale=1):
    """Rescale the Laplacian eigenvalues in [-scale,scale]."""
    M, M = L.shape
    I = sparse.identity(M, format='csr', dtype=L.dtype)
    L *= 2 * scale / lmax
    L -= I
    return L


def build_laplacians(nsides, indexes=None, use_4=False):
    """Build a list of Laplacians (and down-sampling factors) from a list of nsides."""
    L = []
    p = []
    if indexes is None:
        indexes = [None] * len(nsides)
    for i, (nside, index) in enumerate(zip(nsides, indexes)):
        if i > 0:  # First is input dimension.
            p.append((nside_last // nside)**2)
        nside_last = nside
        if i < len(nsides) - 1:  # Last does not need a Laplacian.
            laplacian = healpix_laplacian(nside=nside, indexes=index, use_4=use_4)
            L.append(laplacian)
    # L for residual layers
    L_res = healpix_laplacian(nside=nsides[-1], indexes=indexes[-1], use_4=use_4)
    return L, L_res, p
