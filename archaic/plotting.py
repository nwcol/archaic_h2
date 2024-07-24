"""
Various plotting functions. mostly called by console scripts
"""
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib.ticker as mticker
from matplotlib.lines import Line2D
import numpy as np
import scipy

from archaic import utils


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


"""
Plotting H2
"""


def plot_H2_spectra(
    *args,
    plot_H=True,
    colors=None,
    labels=None,
    n_cols=5,
    alpha=0.05,
    ylim_0=True,
    log_scale=False
):
    # they all have to be the same shape
    if colors is None:
        colors = ['black', 'blue', 'red', 'green']
    # get the confidence interval based on the alpha level
    ci = scipy.stats.norm().ppf(1 - alpha / 2)
    spectrum = args[0]
    n_axs = spectrum.n
    if plot_H:
        if len(args[0].sample_ids) > 1:
            n_axs += 2
            has_H_xy = True
        else:
            n_axs += 1
            has_H_xy = False
    n_rows = int(np.ceil(n_axs / n_cols))
    if n_axs < n_cols:
        n_cols = n_axs
    if log_scale:
        width = 2.8
    else:
        width = 2.2
    fig, axs = plt.subplots(
        n_rows, n_cols, figsize=(n_cols * width, n_rows * 1.8),
        layout="constrained"
    )
    axs = axs.flat
    # get rid of excess subplots
    for ax in axs[n_axs:]:
        ax.remove()
    for i, spectrum in enumerate(args):
        plot_H2_spectrum(
            spectrum, color=colors[i], axs=axs, ci=ci, ylim_0=ylim_0,
            log_scale=log_scale
        )
    if plot_H:
        # required offset
        ax1 = axs[spectrum.n]
        if has_H_xy:
            ax2 = axs[spectrum.n + 1]
        else:
            ax2 = None
        for i, spectrum in enumerate(args):
            plot_H_on_H2_spectrum(
                spectrum, ax1, ax2, color=colors[i], ci=ci, ylim_0=ylim_0,
                log_scale=log_scale
            )
    # write the legend
    if labels is not None:
        y = 1 / n_rows * 0.35
        legend_elements = [
            Line2D([0], [0], color=colors[i], lw=2, label=labels[i])
            for i in range(len(labels))
        ]
        fig.legend(
            handles=legend_elements,
            loc='lower center',
            shadow=True,
            fontsize=10,
            ncols=3,
            bbox_to_anchor=(0.5, -y)
        )
    return fig, axs


def plot_H2_spectrum(
    spectrum,
    color=None,
    axs=None,
    n_cols=5,
    ci=1.96,
    ylim_0=True,
    log_scale=False
):
    #
    if color is None:
        color = 'black'
    if axs is None:
        # if no axs was provided as an argument, create a new one
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
            y_err = np.sqrt(var) * ci
        else:
            y_err = None
        ax = axs[i]
        if spectrum.has_H:
            data = spectrum.data[:-1, i]
        else:
            data = spectrum.data[:, i]
        plot_single_H2(
            ax, x, data, color, y_err=y_err, title=_id, ylim_0=ylim_0,
            log_scale=log_scale
        )
    return 0


def plot_single_H2(
    ax,
    x,
    data,
    color,
    y_err=None,
    title=None,
    ylim_0=True,
    log_scale=False
):
    #
    if y_err is None:
        # we're plotting expectations, with no variance
        ax.plot(x, data, color=color)
    else:
        # we're plotting empirical data with variance
        ax.errorbar(x, data, yerr=y_err, color=color, fmt=".", capsize=0)
    ax.set_xscale('log')
    ax.grid(alpha=0.2)
    if log_scale:
        ax.set_yscale('log')
    else:
        if ylim_0:
            ax.set_ylim(0, )
    if title is not None:
        title = parse_label(title)
        ax.set_title(f'$H_2$ {title}')
    # format the ticks
    format_ticks(ax)
    return 0


def plot_H_on_H2_spectrum(
    spectrum,
    ax1,
    ax2,
    color='black',
    ci=1.96,
    ylim_0=True,
    log_scale=False
):
    #
    ids = spectrum.ids
    one_sample = np.where(ids[:, 0] == ids[:, 1])[0]
    H = spectrum.data[-1, one_sample]
    x1 = np.arange(len(H))
    if spectrum.covs is None:
        ax1.scatter(x1, H, color=color, marker='_')
    else:
        H_var = spectrum.covs[-1, one_sample, one_sample]
        H_y_err = np.sqrt(H_var) * ci
        ax1.errorbar(x1, H, yerr=H_y_err, color=color, fmt='.')
    labels = [parse_label(x) for x in ids[one_sample]]
    ax1.set_xticks(x1, labels, fontsize=9, rotation=60)
    ax1.set_title('$H$')
    axs = [ax1]

    if ax2 is not None:
        two_sample = np.where(ids[:, 0] != ids[:, 1])[0]
        H_xy = spectrum.data[-1, two_sample]
        x2 = np.arange(len(H_xy))
        if spectrum.covs is None:
            ax2.scatter(x2, H_xy, color=color, marker='_')
        else:
            H_xy_var = spectrum.covs[-1, two_sample, two_sample]
            H_xy_y_err = np.sqrt(H_xy_var) * ci
            ax2.errorbar(x2, H_xy, yerr=H_xy_y_err, color=color, fmt='.')
        _labels = [parse_label(x) for x in ids[two_sample]]
        ax2.set_xticks(x2, _labels, fontsize=9, rotation=60)
        ax2.set_title('$H_{xy}$')
        axs.append(ax2)

    for ax in axs:
        ax.grid(alpha=0.2)
        if log_scale:
            ax.set_yscale('log')
        else:
            if ylim_0:
                ax.set_ylim(0, )
    return 0


def parse_label(label):
    # expects population identifiers of form np.array([labelx, labely])
    x, y = label
    if x == y:
        _label = x[:3]
    else:
        _label = f'{x[:3]}-{y[:3]}'
    return _label


def format_ticks(ax):
    # latex scientific notation for x, y ticks
    def scientific(x):
        if x == 0:
            ret = '0'
        else:
            sci_string = np.format_float_scientific(x, precision=2)
            base, power = sci_string.split('e')
            # clean up the strings
            base = base.rstrip('.')
            power = power.lstrip('0')
            if float(base) == 1.0:
                ret = rf'$10^{{{int(power)}}}$'
            else:
                ret = rf'${base} \cdot 10^{{{int(power)}}}$'
        return ret

    formatter = mticker.FuncFormatter(lambda x, p: scientific(x))
    ax.xaxis.set_major_formatter(formatter)
    ax.yaxis.set_major_formatter(formatter)
    return 0


"""
Plotting parameters
"""


def plot_parameters(
    names,
    truths,
    bounds,
    labels,
    *args,
    n_cols=5,
    marker_size=2,
    title=None
):
    # plot parameter clouds
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
            ax.scatter(
                arr[:, i],
                arr[:, j],
                color=colors[z],
                marker='.',
                s=marker_size
            )
        if truths is not None:
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
    if title is not None:
        fig.suptitle(title)
    return 0


def box_plot_parameters(
    pnames,
    truths,
    bounds,
    labels,
    *args,
    n_cols=5,
    title=None
):
    # make box plots comparing distribution of inferred parameters about
    # simulation parameters
    n_axs = len(pnames)
    n_rows = int(np.ceil(n_axs / n_cols))
    fig, axs = plt.subplots(
        n_rows, n_cols, figsize=(n_cols * 2.5, n_rows * 3),
        layout="constrained"
    )
    axs = axs.flat
    for ax in axs[n_axs:]:
        ax.remove()
    colors = ['blue', 'red', 'green']
    for i, ax in enumerate(axs):
        ax.set_title(pnames[i])
        ax.set_ylabel(pnames[i])
        ax.scatter(0, truths[i], marker='x', color='black')
        boxes = ax.boxplot(
            [arr[:, i] for arr in args],
            vert=True,
            patch_artist=True
        )
        for j, patch in enumerate(boxes['boxes']):
            patch.set_facecolor(colors[j])
        ax.set_ylim(bounds[i])
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
    if title is not None:
        fig.suptitle(title)
    return 0


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
    #
    dist = np.histogram(data, bins=bins)[0]
    dist = dist / dist.sum()
    x = bins[1:]
    ax.plot(x, dist, color=color, label=label)
    ax.set_ylabel("freq")
    ax.set_xlim(bins[0], bins[-1])
    ax.grid(alpha=0.2)
    ax.legend(fontsize=8)
    return ax

