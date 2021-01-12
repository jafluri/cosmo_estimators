import matplotlib.pyplot as plt
import healpy as hp
import numpy as np
import seaborn

def plot_n_of_z(n_of_zs):
    """
    This function reproduces the plot of the redshift distributions of the paper in the appendix.
    :param n_of_zs: The redshift distributions of the individual tomographic bins
    """
    plt.figure(figsize=(12, 8))
    palette = seaborn.color_palette("dark")

    for i, n_of_z in enumerate(n_of_zs):
        plt.plot(n_of_z[:, 0], n_of_z[:, 1], label=f"$n_{i}(z)$", c=palette[i])
        plt.fill_between(n_of_z[:, 0], n_of_z[:, 1], color=palette[i], alpha=0.25)
    plt.plot(n_of_z[:, 0], np.sum(n_of_zs, axis=0)[:,1], 'k-', label="$n(z)$")

    plt.yticks([])
    plt.ylim(ymin=0.0)

    plt.xlim(0.0, 3.5)
    plt.xticks(fontsize=15)
    plt.xlabel("$z$", fontsize=20)

    plt.legend(fontsize=20)

def plot_patch(mask, mask_pad, nest=True):
    """
    Plots the mask and the padding.
    :param mask: Boolean mask of the relevant pixels
    :param mask_pad: Boolean mask of the relevant pixels, including padding
    :param nest: True, if the ordering of the masks is NEST
    """
    m = np.ones_like(mask, dtype=float)*hp.UNSEEN
    m[mask_pad] = 0
    m[mask] = 1
    hp.mollview(m, title="Mask and padding of the survey", nest=nest, cbar=False)


