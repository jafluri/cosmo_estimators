import matplotlib
import matplotlib.pyplot as plt
plt.rc("text", usetex=True)

from scipy.stats import linregress
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

def plot_spectra(spectra):
    """
    Plots a list of spectra
    :param spectra: 2d array of size [n, size_specs] of auto and cross spectra ordered like
                    (11, 12, ..., 1n, 22, .., 2n, ..., nn)
    """
    l_spec = len(spectra)
    n_spec = np.maximum(int(0.5*(-1 + np.sqrt(1 + 8*l_spec))), int(0.5*(-1 - np.sqrt(1 + 8*l_spec))))
    fig, ax = plt.subplots(4,4)
    fig.set_size_inches(4*n_spec, 4*n_spec)
    print(ax.shape)
    counter = 0
    for i in range(n_spec):
        for j in range(n_spec):
            if j >= i:
                ax[i,j].loglog(spectra[counter], "k", label=f"z-bins: {i}{j}")
                ax[i,j].grid()
                ax[i,j].set_xlim(xmin=3)
                ax[i, j].set_xlabel("$\ell$", fontsize=15)
                ax[i, j].set_ylabel("$C(\ell)$", fontsize=15)
                ax[i, j].legend(loc="upper right", fontsize=10)
                counter += 1
            else:
                ax[i,j].set_visible(False)
    plt.subplots_adjust(hspace=0.3, wspace=0.3)

def make_map(map_test1, cmap=plt.cm.gray_r, title='', rot=(0, 0, 0),
             dtheta=5, fig=None, sub=None):
    """
    Plots a map with mollview with a square cutout in the middle.
    """
    cmap.set_over('w')
    cmap.set_under('w')
    cmap.set_bad('lightgrey')
    dot_size = 10
    hp.mollview(map_test1, title='{}'.format(title), rot=rot, fig=fig, sub=sub, cmap=cmap)
    hp.graticule();

    lw = 1.5

    theta = np.linspace(90 - dtheta, 90 + dtheta, 100) * np.pi / 180.
    phi = np.linspace(-dtheta, -dtheta, 100) * np.pi / 180.
    hp.projplot(theta, phi, c='k', lw=lw);

    theta = np.linspace(90 - dtheta, 90 + dtheta, 100) * np.pi / 180.
    phi = np.linspace(dtheta, dtheta, 100) * np.pi / 180.
    hp.projplot(theta, phi, c='k', lw=lw);

    theta = np.linspace(90 - dtheta, 90 - dtheta, 100) * np.pi / 180.
    phi = np.linspace(-dtheta, dtheta, 100) * np.pi / 180.
    hp.projplot(theta, phi, c='k', lw=lw);

    theta = np.linspace(90 + dtheta, 90 + dtheta, 100) * np.pi / 180.
    phi = np.linspace(-dtheta, dtheta, 100) * np.pi / 180.
    hp.projplot(theta, phi, c='k', lw=lw);


def make_zoom(map_test1, cmap=plt.cm.gray_r, title='', dtheta=5,
              clim=[-0.025, 0.041], fig=None, sub=None):
    """
    Does the zoom cutput for the <plot_patches_nice> routine
    """
    edge_arcmin = 2 * dtheta * 60
    n_pix = 500
    reso = edge_arcmin / float(n_pix)
    plt.subplot(1, 4, 3)

    hp.gnomview(map_test1, reso=reso, xsize=n_pix, notext=True,
                title=title, rot=(0, 0, 0), cmap=cmap, min=clim[0], max=clim[1],
                hold=True, fig=fig, sub=sub)
    hp.graticule()

def plot_patches_nice(maps, data_mask_pad, fontsize=15, clim=[-0.025, 0.041]):
    """
    Plots a list of patches with a central zoom.
    :param maps: list of maps to plot (only relevant pixels)
    :param data_mask_pad: boolian mask of the survey
    :param fontsize: fontsize to use for the plots
    :param clim: color bar limits for the zoom plots
    """
    matplotlib.rcParams.update({'font.size': fontsize})
    for i, patch in enumerate(maps):
        m = np.full(len(data_mask_pad), hp.UNSEEN)
        m[data_mask_pad] = patch
        m = hp.reorder(m, n2r=True)
        fig = plt.figure(figsize=(16, 8), num=1, constrained_layout=True)
        # cmap = plt.cm.viridis
        # cmap = plt.cm.Blues
        cmap = plt.cm.viridis
        make_zoom(m, cmap=cmap, fig=1, sub=(1, 3, 2), clim=clim)
        make_map(m, cmap=cmap, fig=1, sub=(1, 2, 1), title=f"Patch for z-bin: {i}")

        # borders (up)
        coord1 = [0.529, 0.599]
        coord2 = [0.698, 0.599]
        line = matplotlib.lines.Line2D((coord1[0], coord2[0]), (coord1[1], coord2[1]),
                                       transform=fig.transFigure, c="k")
        fig.lines += [line]

        # down
        coord1 = [0.529, 0.26]
        coord2 = [0.698, 0.26]
        line = matplotlib.lines.Line2D((coord1[0], coord2[0]), (coord1[1], coord2[1]),
                                       transform=fig.transFigure, c="k")
        fig.lines += [line]

        # left
        coord1 = [0.529, 0.599]
        coord2 = [0.529, 0.26]
        line = matplotlib.lines.Line2D((coord1[0], coord2[0]), (coord1[1], coord2[1]),
                                       transform=fig.transFigure, c="k")
        fig.lines += [line]

        # right
        coord1 = [0.698, 0.599]
        coord2 = [0.698, 0.26]
        line = matplotlib.lines.Line2D((coord1[0], coord2[0]), (coord1[1], coord2[1]),
                                       transform=fig.transFigure, c="k")
        fig.lines += [line]

        coord1 = [0.2665, 0.4015]
        coord2 = [0.529, 0.599]
        line = matplotlib.lines.Line2D((coord1[0], coord2[0]), (coord1[1], coord2[1]),
                                       transform=fig.transFigure, c="k")
        fig.lines += [line]

        coord1 = [0.2665, 0.369]
        coord2 = [0.529, 0.26]
        line = matplotlib.lines.Line2D((coord1[0], coord2[0]), (coord1[1], coord2[1]),
                                       transform=fig.transFigure, c="k")
        fig.lines += [line]
        plt.show()

def plot_true_vs_predicted(labels, predictions, param_labels, fiducial_point):
    """
    Plot the mean predictions and their standard deviations against the true labels
    :param labels: 2D array of shape [n_labels, n_params]
    :param predictions: 3D predictions to plot of shape [n_labels, n_predictions, n_params]
    :param param_labels: labels of the parameter
    :param fiducial_point: fiducial point to plot
    """
    # mean and std
    preds_m = np.mean(predictions, axis=1)
    preds_s = np.std(predictions, axis=1)

    # number of params
    n_params = len(param_labels)

    # plot
    plt.figure(figsize=(12,8*n_params))
    for i in range(n_params):
        plt.subplot(n_params, 1, i+1)

        # the x = y line
        min_label = np.min(labels[:,i])
        max_label = np.max(labels[:, i])
        dif_label = max_label - min_label
        x = np.linspace(min_label - 0.05*dif_label, max_label + 0.05*dif_label)
        plt.plot(x, x, linestyle="-", color="k", zorder=1000)

        # the error bars
        plt.errorbar(labels[:,i], preds_m[:,i], yerr=preds_s[:,i], fmt='o', c="#0d47a1", ecolor='lightsteelblue',
                     capsize=5)
        # lin regression
        slope, intercept, _, _, _ = linregress(labels[:,i], preds_m[:,i])
        # plt.plot(x, slope*x + intercept, "k:", zorder=1000)
        plt.axvline(fiducial_point[i], c="k", linestyle=":", linewidth=2)
        plt.xlim(x[0], x[-1])
        plt.ylim(x[0] - 0.05, x[-1] + 0.05)
        plt.xticks(fontsize=15)
        plt.xlabel(r"$%s^\mathrm{true}$" %(param_labels[i]), fontsize=25)
        plt.yticks(fontsize=15)
        plt.ylabel(r"$%s^\mathrm{pred}$" %(param_labels[i]), fontsize=25)
        plt.grid()
