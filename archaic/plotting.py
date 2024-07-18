
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.lines import Line2D
import numpy as np
from archaic import utils


"""
Plotting H2
"""


def plot_H2_spectra(*args, n_cols=5):
    # they all have to be the same shape
    colors = ['black', 'blue', 'red']
    spectrum = args[0]
    n_axs = spectrum.n
    n_rows = int(np.ceil(n_axs / n_cols))
    fig, axs = plt.subplots(
        n_rows, n_cols, figsize=(n_cols * 2, n_rows * 2),
        layout="constrained"
    )
    axs = axs.flat
    for ax in axs[n_axs:]:
        ax.remove()
    for i, spectrum in enumerate(args):
        plot_H2_spectrum(spectrum, color=colors[i], axs=axs)


def plot_H2_spectrum(spectrum, color=None, axs=None, n_cols=5):
    #
    if color is None:
        color = 'black'
    if axs is None:
        n_axs = spectrum.n
        n_rows = int(np.ceil(n_axs / n_cols))
        fig, axs = plt.subplots(
            n_rows, n_cols, figsize=(n_cols * 2, n_rows * 2),
            layout="constrained"
        )
        axs = axs.flat
        for ax in axs[n_axs:]:
            ax.remove()
    x = spectrum.r_bins[:-1] + np.diff(spectrum.r_bins)
    for i, _id in enumerate(spectrum.ids):
        if spectrum.covs is not None:
            if spectrum.has_H:
                var = spectrum.covs[:-1, i, i]
            else:
                var = spectrum.covs[:, i, i]
            y_err = np.sqrt(var) * 1.96
        else:
            y_err = None
        ax = axs[i]
        if spectrum.has_H:
            data = spectrum.data[:-1, i]
        else:
            data = spectrum.data[:, i]
        plot_single_H2(ax, x, data, color, y_err=y_err, title=_id)
    return 0


def plot_single_H2(ax, x, data, color, y_err=None, title=None):
    #
    if y_err is None:
        # we have expectations or something
        ax.plot(x, data, color=color)
    else:
        # we have empirical data with variance
        ax.errorbar(x, data, yerr=y_err, color=color, fmt=".", capsize=0)
    ax.set_xscale('log')
    ax.grid(alpha=0.2)
    ax.set_ylim(0, )
    if title is not None:
        title = parse_label(title)
        ax.set_title(title)
    return 0


def parse_label(label):
    # expects population identifiers of form np.array([labelx, labely])
    x, y = label
    if x == y:
        _label = x
    else:
        _label = f'{x[:3]}-{y[:3]}'
    return _label


"""
Plotting parameters
"""


def plot_parameters(names, truths, bounds, labels, *args, n_cols=5):
    # truths is a vector of underlying true parameters
    n = len(names)
    n_axs = utils.n_choose_2(n)
    n_rows = int(np.ceil(n_axs / n_cols))
    fig, axs = plt.subplots(
        n_rows, n_cols, figsize=(n_cols * 3.5, n_rows * 3),
        layout="constrained"
    )
    axs = axs.flat
    for ax in axs[n_axs:]:
        ax.remove()
    colors = ['blue', 'red', 'green']
    idxs = utils.get_pair_idxs(n)
    for k, (i, j) in enumerate(idxs):
        ax = axs[k]
        ax.set_xlabel(names[i])
        ax.set_ylabel(names[j])
        ax.set_xlim(bounds[i])
        ax.set_ylim(bounds[j])
        for z, arr in enumerate(args):
            ax.scatter(arr[:, i], arr[:, j], color=colors[z], marker='.')
        ax.scatter(truths[i], truths[j], color='black', marker='x')
    legend_elements = [
        Line2D(
            [0],
            [0],
            marker='.',
            color='w',
            label=labels[i],
            markerfacecolor=colors[i],
            markersize=10
        ) for i in range(len(labels))
    ]
    fig.legend(
        handles=legend_elements,
        loc='lower center',
        shadow=True,
        fontsize=10,
        ncols=3,
        bbox_to_anchor=(0.5, -0.1)
    )
    return 0




















"""
Useful constants
"""


_line_styles = [
    "solid",
    "dashed",
    "dotted",
    "dashdot",
    (0, (1, 1)),
    (0, (2, 2)),
    (0, (1, 2, 2, 2)),
    (0, (3, 4, 2, 4))
]


_colors = [
    "red",
    "blue",
    "green"
]


"""
Colors and color maps
"""


def get_gnu_cmap(n):

    cmap = list(cm.gnuplot(np.linspace(0, 0.95, n)))
    return cmap


def get_terrain_cmap(n):

    cmap = list(cm.terrain(np.linspace(0, 0.95, n)))
    return cmap


"""
Plotting graph statistics
"""


def plot_curves(axs, H, H2, r, sample_names, pair_names, color, log_scale,
                label=None):

    shape = axs.shape
    offset = 1
    n = len(sample_names)
    plot_H(axs[0, 0], H[:n], sample_names, color, label=label, title="$H$")
    if len(pair_names) > 0:
        offset += 1
        plot_H(axs[0, 1], H[n:], pair_names, color, title="$H_{xy}$")
    for i in range(len(sample_names)):
        idx = np.unravel_index(i + offset, shape)
        title = f"$H_2$:{sample_names[i]}"
        plot_H2(axs[idx], r, H2[:, i], color, log_scale=log_scale, title=title)
    offset += n
    for i in range(len(pair_names)):
        idx = np.unravel_index(i + offset, shape)
        title = f"$H_{{2,xy}}$:{pair_names[i]}"
        plot_H2(axs[idx], r, H2[:, i + n], color, log_scale=log_scale, title=title)
    return 0


def plot_error_points(axs, H, H_err, H2, H2_err, r, sample_names, pair_names,
                      color, log_scale, label=None):

    shape = axs.shape
    offset = 1
    n = len(sample_names)
    plot_H_err(axs[0, 0], H[:n], H_err[:n], sample_names, color, label=label, title="$H$")
    if len(pair_names) > 0:
        offset += 1
        plot_H_err(axs[0, 1], H[n:], H_err[n:], pair_names, color, title="$H_{xy}$")
    for i in range(len(sample_names)):
        idx = np.unravel_index(i + offset, shape)
        title = f"$H_2$:{sample_names[i]}"
        plot_H2_err(axs[idx], r, H2[:, i], H2_err[:, i], color, log_scale=log_scale, title=title)
    offset += len(sample_names)
    for i in range(len(pair_names)):
        idx = np.unravel_index(i + offset, shape)
        title = f"$H_{{2,xy}}$:{pair_names[i]}"
        plot_H2_err(axs[idx], r, H2[:, i + n], H2_err[:, i + n], color, log_scale=log_scale, title=title)
    return 0


"""
Plotting functions dedicated to statistics
"""


def plot_H(ax, H, names, color, label=None, title=None):

    for i, H in enumerate(H):
        if i >= 1:
            label = None
        ax.scatter(i, H, color=color, marker="_", label=label)
    ax.set_xticks(np.arange(len(names)), names)
    ax.grid(alpha=0.2)
    if title:
        ax.set_title(title)


def plot_H_err(ax, H, H_err, names, color, label=None, title=None):

    for i, H in enumerate(H):
        if i >= 1:
            label = None
        ax.errorbar(i, H, yerr=H_err[i], color=color, fmt='.', label=label)
    ax.set_xticks(np.arange(len(names)), names)
    ax.grid(alpha=0.2)
    if title:
        ax.set_title(title)


def _plot_H_err(ax, H, H_err, E_H, names, colors, E_colors, title=None):

    abbrev_names = []
    for i, name in enumerate(names):
        if type(name) == str:
            name = name[:3]
        else:
            name = f"{name[0][:3]}-{name[1][:3]}"
        abbrev_names.append(name)
        ax.errorbar(i, H[i], yerr=H_err[i], color=colors[i], fmt='.')
        ax.scatter(i, E_H[i], color=E_colors[i], marker='+')
    ax.set_ylim(0, )
    ax.set_xticks(np.arange(len(names)), abbrev_names)
    ax.grid(alpha=0.2)
    if title:
        ax.set_title(title)
    return ax


def plot_H2(ax, r, H2, color, log_scale=False, title=None):
    # for plotting expectations
    ax.plot(r, H2, color=color)
    ax.set_xscale("log")
    ax.autoscale()
    ax.grid(alpha=0.2)
    if log_scale:
        ax.set_yscale("log")
    if title:
        ax.set_title(title)
    return ax


def plot_H2_err(ax, r, H2, H2_err, color, log_scale=False, title=None):
    # for plotting empirical values
    ax.errorbar(r, H2, yerr=H2_err, color=color, fmt=".", capsize=0)
    ax.set_xscale("log")
    ax.grid(alpha=0.2)
    if log_scale:
        ax.set_yscale("log")
    else:
        ax.set_ylim(0, )
    if title:
        ax.set_title(title)
    return ax


def _plot_H2_err(ax, r, H2, H2_err, E_H2, color, E_color, log_scale=False,
                title=None):

    ax.errorbar(
        r, H2, yerr=H2_err, color=color, fmt=".", capsize=0
    )
    ax.plot(r, E_H2, color=E_color)
    ax.set_xscale("log")
    if log_scale:
        ax.set_yscale("log")
    ax.grid(alpha=0.2)
    if title:
        ax.set_title(title)
    return ax


def _plot_H2_err(ax, r, H2, names, colors, fmts=None, y_errs=None, log_scale=False, y_lim=None, title=None):
    # plot several H2 curves. H2 is expected to be shape (n, r) array
    if H2.ndim == 1:
        n = 1
    else:
        n = len(H2)
    if fmts == None:
        fmts = ["-"] * n
    else:
        if type(fmts) != list:
            fmts = [fmts] * n
        elif len(fmts) < n:
            fmts = [fmts[0]] * n
    if y_errs == None:
        y_errs = [None] * n

    for i, name in enumerate(names):
        ax.errorbar(
            r, H2[i], yerr=y_errs[i], color=colors[i], fmt=fmts[i], label=name
        )
    ax.set_xscale("log")
    ax.set_ylabel("$H_2$")
    ax.set_xlabel("r")
    ax.grid(alpha=0.2)
    if log_scale:
        ax.set_yscale("log")
    else:
        if y_lim:
            ax.set_ylim(0, y_lim)
        else:
            ax.set_ylim(0, )
    if title:
        ax.set_title(title)
    return ax


"""
Generic plotting functions
"""


def plot_distribution(
    bins,
    data,
    ax,
    color="black",
    label=None
):

    distribution = np.histogram(data, bins=bins)[0]
    distribution = distribution / distribution.sum()
    x = bins[1:]
    ax.plot(x, distribution, color=color, label=label)
    ax.set_ylabel("freq")
    ax.set_xlim(bins[0], bins[-1])
    ax.grid(alpha=0.2)
    ax.legend(fontsize=8)
    return ax

