
from bokeh.palettes import Category10, Turbo256, TolRainbow
import matplotlib as mpl
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import scipy
from scipy import stats

from h2py import util, h2_parsing, inference


mpl.rcParams['text.usetex'] = True
mpl.rcParams['text.latex.preamble'] = \
    "\\usepackage{amsmath}\\usepackage{amssymb}"
mpl.rcParams['font.family'] = "serif"
mpl.rcParams['font.serif'] = "Computer Modern"
mpl.rcParams['savefig.bbox'] = "tight"


def plot_H2s(
    models=None, 
    datas=None,
    labels=None,
    conf=0.95,
    plot_H=True,
    ylim_0=False
):
    """
    Plot one or more sets of H2 curves.
    """
    ci = stats.norm().ppf(0.5 + conf / 2)
    width_fac = 0.09
    height_ratio = 1.1
    n_cols = 5

    if len(datas) > 0:
        data = datas[0]
        num_stats = data['means'].shape[1]
        bins = data['bins']
    else:
        model = models[0]
        num_stats = model['means'].shape[1]
        bins = h2_parsing._default_bins

    if plot_H:
        n_axs = num_stats + 2
    else:
        n_axs = num_stats

    n_rows = int(np.ceil(n_axs / n_cols))
    if n_axs < n_cols:
        n_cols = n_axs
    width = width_fac * len(bins)
    height = width / height_ratio

    fig, axs = plt.subplots(
        n_rows, 
        n_cols, 
        figsize=(n_cols * width, n_rows * height), 
        layout="constrained"
    )
    axs = axs.flat
    for ax in axs[n_axs:]:
        ax.remove()

    if len(datas) + len(models) == 1:
        colors = ['black']
    elif len(datas) == 1 and len(models) == 1:
        colors = ['black', 'black'] 
    elif len(datas) + len(models) == 2:
        colors = [TolRainbow[3][0], TolRainbow[3][2]]
    elif len(datas) + len(models) <= 11:
        colors = TolRainbow[len(datas) + len(models)]
    else:
        gap = int(np.floor(256 / len(datas) + len(models)))
        colors = [Turbo256[i] for i in range(0, 255, gap)]

    lines = []
    for data, color in zip(datas + models, colors):
        l = plot_H2(
            data,
            axs,
            color,
            ci,
            n_axs,
            two_sample=True
        )
        lines.append(l)

    n_legend_rows = int(np.ceil(len(labels) / 4))
    y_anchor = (-0.175 * n_legend_rows) / n_rows

    plt.figlegend(
        lines, 
        labels, 
        framealpha=0,
        loc='lower center',
        bbox_to_anchor=(0.5, y_anchor),
        ncols=3
    )

    for i, ax in enumerate(axs):
        if ax is None:
            continue
        if i in [n_axs - 1, n_axs - 2]: 
            pass
            #ax.set_ylabel('$H$')
            #format_ticks(ax, x_ax=False)
        else:
            ax.set_xscale('log')
            ax.set_xlim(max(1e-8, 0.5 * bins[0]), 2 * bins[-1])
            #format_ticks(ax)
        if ylim_0:
            ax.set_ylim(0, )
    
    return


def plot_H2(
    data,
    axs,
    color,
    ci,
    n_axs,
    two_sample=True,
    plot_H=True
):

    empirical = {
        'fmt': 'o',
        'markersize': 3.7,
        'elinewidth': 0.7,  
        'markerfacecolor': 'none',
        'markeredgecolor': color, 
        'markeredgewidth': 0.7,
        'ecolor': color,
        'capsize': 0
    }

    exp = {
        'linewidth': 0.8,
    }
    exp_scatter = {
        'marker': 'x',
        's': 16,
        'linewidth': 0.7
    }


    hlabels = []
    hxylabels = []
    x = data['bins'][:-1] + np.diff(data['bins']) / 2
    k = 0
    for i, sample_i in enumerate(data['pop_ids']):
        for sample_j in data['pop_ids'][i:]:
            if sample_i == sample_j: 
                label = sample_i  
                hlabels.append(label[:3])
                Hax = n_axs - 2
                hpos = i
            else:
                if two_sample:
                    label = f'{sample_i[:3]},{sample_j[:3]}'
                    hxylabels.append(label)
                    Hax = n_axs - 1
                    hpos = k - i - 1
                else:
                    continue

            H = data['means'][-1, k]
            H2 = data['means'][:-1, k]

            if 'covs' in data:
                stdH2 = data['covs'][:-1, k, k] ** 0.5 * ci
                stdH = data['covs'][-1, k, k] ** 0.5 * ci
                l,*_ = axs[k].errorbar(
                    x, 
                    H2, 
                    yerr=stdH2, 
                    **empirical
                )
                if plot_H:
                    axs[Hax].errorbar( 
                        hpos, 
                        H, 
                        yerr=stdH, 
                        **empirical
                    )

            else:
                l, = axs[k].plot(x, H2, color=color, **exp)
                if plot_H:
                    axs[Hax].scatter(hpos, H, color=color, **exp_scatter)

            axs[k].set_title(label)
            k += 1

    if plot_H:
        axs[n_axs - 2].set_xticks(
            np.arange(len(hlabels)), labels=hlabels, rotation=90, fontsize=8
        )
        axs[n_axs - 1].set_xticks(
            np.arange(len(hxylabels)), labels=hxylabels, rotation=90, fontsize=8
        )
    return l


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
    return


def plot_params(
    names,
    p_arr,
    lls,
    n_cols=5,
    title=None,
):
    """
    Make pairwise scatter plots of parameters.
    """
    n = len(names)
    n_axes = n * (n - 1) // 2
    n_rows = int(np.ceil(n_axes / n_cols))

    fig, axes = plt.subplots(
        n_rows, 
        n_cols, 
        figsize=(n_cols * 2.4, n_rows * 2.3),
        layout="constrained"
    )
    axes = axes.flat
    for ax in axes[n_axes:]:
        ax.remove()

    idxs = [(i, j) for i in range(n - 1) for j in range(i + 1, n)]
    norm = plt.Normalize(np.min(lls), np.max(lls))

    for ax, (i, j) in zip(axes, idxs):
        ax.set_xlabel(names[i])
        ax.set_ylabel(names[j])
        cb = ax.scatter(
            p_arr[:, i],
            p_arr[:, j],
            c=lls,
            cmap='plasma',
            marker='o',
            s=12,
            norm=norm
        )
        rho = stats.pearsonr(p_arr[:, i], p_arr[:, j]).statistic
        ax.set_title(
            rf'$\rho={np.round(rho, 2)}$', y=1.0, pad=-14
        )

    smap = plt.cm.ScalarMappable(cmap='plasma', norm=norm)
    cbar = fig.colorbar(
        cb, 
        ax=axes[:n_axes], 
        location='bottom',
        shrink=0.35,
        fraction=0.1
    )
    cbar.ax.tick_params()
    cbar.ax.set_ylabel('ll', rotation=0)

    return




































"""
Plotting H2
"""


def plot_curve(bins, statistic, ax=None, color='black'):
    #
    if not ax:
        fig, ax = plt.subplots(figsize=(6, 4), layout='constrained')
    ax.plot(bins[1:], statistic, color=color)
    ax.set_ylim(0, )
    ax.set_xscale('log')
    ax.set_xlabel('$r$')
    format_ticks(ax)
    return ax


def plot_H2_spectra(
    *args,
    plot_H=True,
    colors=None,
    labels=None,
    n_cols=4,
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
        width = 3 #2.2
    fig, axs = plt.subplots(
        n_rows, n_cols, figsize=(n_cols * width, n_rows * 2.5), #1.8
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
                ax.set_xlim(xlim, 1)

    # write the legend
    if labels is not None:
        y = 0.2
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
            _label = x
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
