import healpy as hp
import numpy as np
import pyccl as ccl

from . import utils

def gen_ccl_spectra(cosmo, n_of_zs, l_max=1001):
    """
    Generates the theoretical weak lensing power spectra for a given cosmology.
    :param cosmo: 1D array of cosmological parameters ordered as (Om, Ob, h, ns, sigma8, w0)
    :param n_of_zs: 3D array of redshift distributions. The first axis enumerates the different distributions, the
                    second the redshift values and the last the actual number counts
    :param l_max: maximum l value to calculate
    :return: The theoretical power spectra of the survey and all cross spectra, the ordering is
             (11, 12, ..., 1n, 22, .., 2n, ..., nn)
    """
    # cosmo needs to be double
    cosmo = cosmo.astype(np.float64)
    # get the ccl
    cosmo = ccl.Cosmology(Omega_c=cosmo[0] - cosmo[1],
                          Omega_b=cosmo[1],
                          h=cosmo[2],
                          n_s=cosmo[3],
                          sigma8=cosmo[4],
                          w0=cosmo[5])

    # Create objects to represent tracers of the weak lensing signal with this
    # number density (with has_intrinsic_alignment=False)
    tracer = []
    for i in range(4):
        tracer.append(ccl.WeakLensingTracer(cosmo, dndz=(n_of_zs[i][:, 0], n_of_zs[i][:, 1])))

    # Calculate the angular cross-spectrum of the two tracers as a function of ell
    print("Calculating spectra...", flush=True)
    ell = np.arange(2, l_max)
    all_cl = []
    for i in range(4):
        for j in range(4):
            if j >= i:
                cl = ccl.angular_cl(cosmo, tracer[i], tracer[j], ell)

                # append zeros
                cl = np.concatenate([np.zeros(2), cl])

                # append no binning
                all_cl.append(cl)

    # stack
    cl = np.stack(all_cl, axis=0)
    return cl

def create_GRF_samples(spectra, data_mask, data_mask_pad, seed, pixwin=False, fwhm=0.0, lmax=1000,
                       return_fidu_spec=False, verbose=0):
    """
    Creates a sample of GRF maps of given spectra that can be saved for later training.
    :param spectra: A list of power spectra used to generate the GRF samples, if the shape of the spectra is [n_bins, N]
                    a tomographic sample is generated and the returned samples have the shape [N_pix, n_bins]
    :param data_mask: a boolian array (or int) representing the observational mask on the sphere
    :param data_mask_pad: a boolian mask of the observation includeing padding
    :param seed: the random seed for ALL samples
    :param pixwin: Convolve alm with pixel window function (default: False)
    :param fwhm: Full width half maximum of the Gaussian smoothing applied to each sample, defaults to no smoothing
    :param lmax: The maximum l used to generate the maps (default: 1000)
    :param return_fidu_spec: return the numerically measured Cl of the map generated with spectra[0]
    :param verbose: verbosity parameter forwarded to synfast, defaults to no output
    :return: A list of maps the same length as spectra each containing sum(data_mask_pad) entries, if return_fidu_spec is
             True, the power and cross spectra of the first entry in spectra will be returned.
    """
    # get the nside
    nside = hp.npix2nside(len(data_mask))

    # invert the data mask
    inv_mask = np.logical_not(data_mask)
    ext_indices = np.arange(len(inv_mask))[data_mask_pad]

    # cycle
    maps_out = []
    for num, spectrum in enumerate(spectra):
        # set seed
        np.random.seed(seed)

        # non tomographic case
        if spectrum.ndim == 1:
            # get map
            m = hp.synfast(cls=spectrum, fwhm=fwhm, nside=nside, pixwin=pixwin, lmax=lmax, verbose=verbose)

            # reorder
            m = hp.reorder(map_in=m, r2n=True)

            # set sourrounding to zero
            m[inv_mask] = 0.0

            # get the measurement if necessary
            if return_fidu_spec and num == 0:
                m_ring = hp.reorder(map_in=m, n2r=True)
                cl_fidu = hp.anafast(m_ring)

            # append only wanted values
            maps_out.append(m[ext_indices])
        # tomographic case
        else:
            # get the number of bins
            n_bins = (-1 + int(np.sqrt(1 + 8 * len(spectrum)))) // 2
            if n_bins * (n_bins + 1) // 2 != len(spectrum):
                raise ValueError("The number of spectra does not seem to be valid!")

            # generate the maps following Raphael's paper
            T_ij_s = []
            for i in range(n_bins):
                for j in range(n_bins):
                    if i == j:
                        index = utils.ij_to_list_index(i, j, n_bins)
                        T_ij = spectrum[index].copy()
                        for k in range(j):
                            index = utils.ij_to_list_index(k, i, n_bins)
                            T_ij -= T_ij_s[index] ** 2
                        T_ij_s.append(np.sqrt(T_ij))
                    elif j > i:
                        index = utils.ij_to_list_index(i, j, n_bins)
                        T_ij = spectrum[index].copy()
                        for k in range(i):
                            index_1 = utils.ij_to_list_index(k, j, n_bins)
                            index_2 = utils.ij_to_list_index(k, i, n_bins)
                            T_ij -= T_ij_s[index_1] * T_ij_s[index_2]
                        index = utils.ij_to_list_index(i, i, n_bins)
                        # set division through 0 to 0
                        T_ij = np.divide(T_ij, T_ij_s[index], out=np.zeros_like(T_ij),
                                         where=T_ij_s[index] != 0)
                        T_ij_s.append(T_ij)

            # now we generate the maps with the right states
            T_ij_maps = []
            counter = 0
            for i in range(n_bins):
                current_state = np.random.get_state()
                for j in range(n_bins):
                    if j >= i:
                        np.random.set_state(current_state)
                        m = hp.synfast(cls=T_ij_s[counter] ** 2, fwhm=fwhm, nside=nside,
                                       pixwin=pixwin, lmax=lmax, verbose=verbose)
                        T_ij_maps.append(m)
                        counter += 1

            # list for maps
            maps = []

            # and now the output maps
            for i in range(n_bins):
                m = np.zeros(hp.nside2npix(nside=nside))
                for j in range(n_bins):
                    if i >= j:
                        index = utils.ij_to_list_index(j, i, n_bins)
                        m += T_ij_maps[index]

                # reorder
                m = hp.reorder(map_in=m, r2n=True)

                # set sourrounding to zero
                m[inv_mask] = 0.0

                # append only wanted values
                maps.append(m[ext_indices])

            # calculate all spectra and cross spectra
            if num == 0 and return_fidu_spec:
                # get the alms
                alms = []
                for i in range(n_bins):
                    # make the map
                    m = np.zeros(hp.nside2npix(nside=nside))
                    m[ext_indices] = maps[i]
                    # reorder and alm
                    m = hp.reorder(m, n2r=True)
                    alms.append(hp.map2alm(m))

                # get the cl
                cl_fidu = []
                for i in range(n_bins):
                    for j in range(n_bins):
                        if j >= i:
                            cl_fidu.append(hp.alm2cl(alms1=alms[i], alms2=alms[j]))

                # stack
                cl_fidu = np.stack(cl_fidu, axis=-1)

            # stack
            maps_out.append(np.stack(maps, axis=-1))

    # return the maps
    if return_fidu_spec:
        return maps_out, cl_fidu

    else:
        return maps_out
