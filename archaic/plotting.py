"""
Various plotting functions. mostly called by console scripts
"""
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib.ticker as mticker
from matplotlib.lines import Line2D
import moments
import numpy as np
import scipy

from archaic import utils, parsing


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
    xlim=None,
    log_scale=False,
    sci=True,
    statistic='$H_2',
    plot_two_sample=True,
    ratio_yticks=False
):
    # they all have to be the same shape
    if colors is None:
        colors = ['black', 'blue', 'red', 'green']
    # get the confidence interval based on the alpha level
    ci = scipy.stats.norm().ppf(1 - alpha / 2)
    spectrum = args[0]

    if plot_two_sample:
        n_axs = spectrum.n
    else:
        n_axs = 10

    if plot_H:
        n_axs += 1
        if len(spectrum.sample_ids) > 1:
            if plot_two_sample:
                n_axs += 1

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
            spectrum,
            color=colors[i],
            axs=axs,
            ci=ci,
            ylim_0=ylim_0,
            log_scale=log_scale,
            plot_H=plot_H,
            sci=sci,
            statistic=statistic,
            plot_two_sample=plot_two_sample
        )
    # adjust ylim etc
    for i, ax in enumerate(axs):
        if ax is None:
            continue
        if ratio_yticks:
            if n_axs - i > 1 + plot_two_sample:
                ax.set_yticks([1, 2])
        ax.grid(alpha=0.2)
        if log_scale:
            ax.set_yscale('log')
        else:
            if ylim_0:
                ax.set_ylim(0, )
            if xlim:
                ax.autoscale_view()
                ax.set_xlim(xlim, )

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
    log_scale=False,
    plot_H=True,
    sci=True,
    statistic='$H_2$',
    plot_two_sample=True
):
    #
    if color is None:
        color = 'black'
    if axs is None:

        # if no axs was provided as an argument, create a new one
        n_axs = spectrum.n

        # we need extra axes if we want to plot H
        if plot_H:
            if len(spectrum.sample_ids) > 1:
                n_axs += 2
            else:
                n_axs += 1

        n_rows = int(np.ceil(n_axs / n_cols))
        fig, axs = plt.subplots(
            n_rows, n_cols, figsize=(n_cols * 2, n_rows * 2),
            layout="constrained"
        )
        axs = axs.flat
        for ax in axs[n_axs:]:
            ax.remove()

        for ax in axs:
            ax.grid(alpha=0.2)
            if log_scale:
                ax.set_yscale('log')
            else:
                if ylim_0:
                    ax.set_ylim(0, )

    # plot H2
    x = spectrum.r_bins[:-1] + np.diff(spectrum.r_bins)
    k = 0
    for i, _id in enumerate(spectrum.ids):
        if not plot_two_sample:
            if _id[0] != _id[1]:
                continue
        if spectrum.covs is not None:
            if spectrum.has_H:
                var = spectrum.covs[:-1, i, i]
            else:
                var = spectrum.covs[:, i, i]
            y_err = np.sqrt(var) * ci
        else:
            y_err = None
        ax = axs[k]
        if spectrum.has_H:
            data = spectrum.data[:-1, i]
        else:
            data = spectrum.data[:, i]
        plot_single_H2(
            ax, x, data, color, y_err=y_err, title=_id, sci=sci,
            statistic=statistic
        )
        k += 1

    # plot H
    ax2 = None
    if plot_H:
        if spectrum.has_H:
            if len(spectrum.sample_ids) > 1:
                if plot_two_sample:
                    ax2 = axs[k + 1]
                    ax1 = axs[k]
                else:
                    ax1 = axs[k]
            else:
                ax1 = axs[k]
            plot_H_on_H2_spectrum(
                spectrum, ax1, ax2, color=color, ci=ci,
            )
    return 0


def plot_single_H2(
    ax,
    x,
    data,
    color,
    y_err=None,
    title=None,
    sci=True,
    statistic='$H_2$'
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
    if title is not None:
        title = parse_label(title)
        ax.set_title(f'{statistic} {title}')
    # format the ticks
    if sci:
        format_ticks(ax)
    return 0


def plot_H_on_H2_spectrum(
    spectrum,
    ax1,
    ax2,
    color='black',
    ci=1.96
):
    #
    ids = spectrum.ids
    if len(ids[0]) == 2:
        one_sample = np.where(ids[:, 0] == ids[:, 1])[0]
    else:
        one_sample = np.arange(len(ids))
        ax2 = None
    H = spectrum.data[-1, one_sample]
    x1 = np.arange(len(H))
    if spectrum.covs is None:
        ax1.scatter(x1, H, color=color, marker='_')
    else:
        H_var = spectrum.covs[-1, one_sample, one_sample]
        H_y_err = np.sqrt(H_var) * ci
        ax1.errorbar(x1, H, yerr=H_y_err, color=color, fmt='.')
    labels = [parse_label(x) for x in ids[one_sample]]
    ax1.set_xticks(x1, labels, fontsize=8, rotation=90)
    ax1.set_title('$H$')

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
        ax2.set_xticks(x2, _labels, fontsize=8, rotation=90)
        ax2.set_title('$H_{xy}$')

    return 0


def parse_label(label):
    # expects population identifiers of form np.array([labelx, labely])
    if len(label) == 2:
        x, y = label
        if x == y:
            _label = x[:3]
        else:
            _label = f'{x[:3]}-{y[:3]}'
    else:
        _label = label
    return _label


def format_ticks(ax, y_ax=True, x_ax=True):
    # latex scientific notation for x, y ticks
    def scientific(x):
        if x == 0:
            ret = '0'
        else:
            sci_string = np.format_float_scientific(x, precision=2)
            base, power = sci_string.split('e')
            # clean up the strings
            base = base.rstrip('0').rstrip('.').rstrip('0')
            power = power.lstrip('0')
            if float(base) == 1.0:
                ret = rf'$10^{{{int(power)}}}$'
            else:
                ret = rf'${base} \cdot 10^{{{int(power)}}}$'
        return ret

    formatter = mticker.FuncFormatter(lambda x, p: scientific(x))
    if x_ax:
        ax.xaxis.set_major_formatter(formatter)
    if y_ax:
        ax.yaxis.set_major_formatter(formatter)
    return 0


"""
publication-quality H2 plot
"""


def plot_two_panel_H2(model, data, labels, colors, axs=None, ci=1.96):
    # labels is a list of strings naming each individual in proper order

    if axs is None:
        fig, (ax0, ax1) = plt.subplots(2, 1, figsize=(3.5, 5), layout='constrained')
    else:
        ax0, ax1 = axs

    sample_ids = data.sample_ids
    if labels is None:
        labels = sample_ids
    names = dict(zip(sample_ids, labels))
    r_bins = data.r_bins
    x = r_bins[:-1] + np.diff(r_bins) / 2

    for i, (id_x, id_y) in enumerate(data.ids):
        H2 = data.arr[:-1, i]
        H = data.arr[-1, i]
        EH2 = model.arr[:-1, i]
        EH = data.arr[-1, i]

        var = data.covs[:-1, i, i]
        y_err = np.sqrt(var) * ci

        if id_x == id_y:
            ax = ax0
            label = names[id_x]
        else:
            ax = ax1
            label = f'{names[id_x]}-{names[id_y]}'
        ax.errorbar(
            x, H2, yerr=y_err, color=colors[i], fmt=".", capsize=0
        )
        ax.plot(x, EH2, color=colors[i], label=label)


    for ax in (ax0, ax1):
        ax.set_ylim(0, )
        ax.set_xlabel('$r$')
        ax.set_ylabel('$H_2$')
        format_ticks(ax)
        ax.set_xscale('log')
        ax.set_xlim(8e-7, 1.15e-2)
        plt.minorticks_off()

    #plt.savefig(out_fname, format='svg', bbox_inches='tight')


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
    title=None,
    wide_bounds=False
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
    colors = ['b', 'orange', 'g', 'r']
    idxs = utils.get_pair_idxs(n)
    for k, (i, j) in enumerate(idxs):
        ax = axs[k]
        ax.set_xlabel(names[i])
        ax.set_ylabel(names[j])
        if wide_bounds:
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
    colors = ['b', 'orange', 'g', 'r']
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
plotting the SFS [using functions from moments]
"""


def plot_SFS():
    #


    return 0


def plot_SFS_residuals():


    return 0


"""
Generic plotting functions
"""


def plot_pair_counts(H2_dict):
    #
    fig, ax = plt.subplots(layout='constrained')
    x = H2_dict['r_bins'][1:]
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.plot(x, H2_dict['n_site_pairs'].sum(0), color='black', label='site_pairs')
    for i, ids in enumerate(H2_dict['ids']):
        if ids[0] == ids[1]:
            ax.plot(
                x, H2_dict['H2_counts'][:, i].sum(0), label=ids[0]
            )
    ax.legend()


def plot_arm_H2_H(H2_dict, idx):
    # assumes that each arm is one window
    tot_H2 = H2_dict['H2_counts'].sum(0) / H2_dict['n_site_pairs'].sum(0)
    left_H2 = H2_dict['H2_counts'][0] / H2_dict['n_site_pairs'][0]
    right_H2 = H2_dict['H2_counts'][1] / H2_dict['n_site_pairs'][1]

    cross_arm_num_h2, cross_arm_num_pairs = parsing.compute_cross_arm_H2(H2_dict, 1)
    cross_arm_H2 = cross_arm_num_h2 / cross_arm_num_pairs

    tot_H_squared = (H2_dict['H_counts'].sum(0) / H2_dict['n_sites'].sum(0)) ** 2
    left_H_squared = (H2_dict['H_counts'][0] / H2_dict['n_sites'][0]) ** 2
    right_H_squared = (H2_dict['H_counts'][1] / H2_dict['n_sites'][1]) ** 2

    fig, ax = plt.subplots(layout='constrained')
    r = H2_dict['r_bins'][1:]

    ax.plot(r, tot_H2[idx], color='black', label='total $H_2$')
    ax.plot(r, left_H2[idx], color='red', label='left-arm $H_2$')
    ax.plot(r, right_H2[idx], color='orange', label='right-arm $H_2$')

    ax.scatter(1, tot_H_squared[idx], marker='x', label='total $H^2$', color='black')
    ax.scatter(1, left_H_squared[idx], marker='x', label='left-arm $H^2$',
               color='red')
    ax.scatter(1, right_H_squared[idx], marker='x', label='right-arm $H^2$',
               color='orange')
    ax.scatter(1, cross_arm_H2[idx], marker='+', label='cross-arm $H_2$', color='black')
    ax.set_xscale('log')
    ax.set_ylim(0, )
    ax.legend()


def plot_H2_vs_Hsquared(dic):
    #
    fig, ax = plt.subplots(layout='constrained')
    bins = dic['r_bins']
    colors = list(cm.gnuplot(np.linspace(0.1, 0.95, 10)))

    for i in range(17, 27, 1):
        bin0 = np.format_float_scientific(bins[i], 2)
        bin1 = np.format_float_scientific(bins[i + 1], 2)
        label = rf'$r\in[{bin0}, {bin1}]$'
        plt.scatter(
            range(55), dic['H2'][:, i], label=label, facecolors='none',
            edgecolors=colors[i-17]
        )

    H_squared = dic['H'] ** 2
    plt.scatter(range(55), H_squared, marker='+', color='black', label='$H^2$')
    plt.ylim(0, )
    fig.legend(fontsize=8, draggable=True)


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



chrs = [np.load(
    f'/home/nick/Projects/archaic/statistics/arms/arm_H2_{i}.npz'
) for i in range(1, 23)]
mean = parsing.sum_H2(*chrs)
