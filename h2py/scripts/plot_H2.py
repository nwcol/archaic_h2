"""
Plot an arbitrary number of H, H2 expectations alongside 0 or 1 empirical vals.
"""
import argparse
from bokeh.palettes import Category10
import demes
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
from scipy import stats
import os

from h2py import h2stats_mod, inference
from h2py.h2stats_mod import H2stats


colors = Category10[10]


def get_args():
    # get args
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-d', '--data_fnames', 
        nargs='*',
        default=[],
        type=str
    )
    parser.add_argument(
        '-g', '--graph_fnames', 
        nargs='*',
        default=[],
        type=str
    )
    parser.add_argument('-o', '--out_fname', required=True)
    parser.add_argument('-u', '--u', type=float, default=None)
    return parser.parse_args()


def plot_H2stats(
    h2stats,
    axs,
    color,
    ci,
    n_axs,
    two_sample=True
):

    hlabels = []
    hxylabels = []
    x = h2stats.bins[:-1] + np.diff(h2stats.bins) / 2
    k = 0
    for i, sample_i in enumerate(h2stats.pop_ids):
        for sample_j in h2stats.pop_ids[i:]:
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

            H = h2stats.stats[-1, k]
            H2 = h2stats.stats[:-1, k]

            if h2stats.covs is not None:
                stdH2 = h2stats.covs[:-1, k, k] ** 0.5 * ci
                stdH = h2stats.covs[-1, k, k] ** 0.5 * ci
                l,*_ = axs[k].errorbar(
                    x, 
                    H2, 
                    yerr=stdH2, 
                    markerfacecolor='none',
                    markeredgecolor=color, 
                    ecolor=color,
                    fmt="o", 
                    capsize=0
                )
                axs[Hax].errorbar( 
                    hpos, 
                    H, 
                    yerr=stdH, 
                    markerfacecolor='none',
                    markeredgecolor=color, 
                    ecolor=color,
                    fmt="o", 
                    capsize=0
                )

            else:
                l, = axs[k].plot(x, H2, color=color)
                axs[Hax].scatter(hpos, H, color=color, marker='x')

            axs[k].set_title(label)
            k += 1
    axs[n_axs - 2].set_xticks(np.arange(len(hlabels)), labels=hlabels, rotation=90)
    axs[n_axs - 1].set_xticks(np.arange(len(hxylabels)), labels=hxylabels, rotation=90)
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


def main():

    args = get_args()
    if len(args.graph_fnames) > 0:
        if args.u is not None:
            u = args.u
        else:
            opt_u = None
            for gf in args.graph_fnames:
                g = demes.load(gf)
                if 'opt_info' in g.metadata:
                    opt_u = g.metadata['opt_info']['u']
            if opt_u is None:
                raise ValueError('please provide a u parameter')
            else:
                u = opt_u
    template = args.graph_fnames[0] if len(args.graph_fnames) > 0 else None
    datas = [H2stats.from_file(f, graph=template) for f in args.data_fnames]
    
    if len(datas) > 0:
        template = datas[0]
        n_stats = template.stats.shape[1]
        bins = template.bins
        models = [H2stats.from_demes(g, u, template=template) 
                for g in args.graph_fnames]
        lls = [inference.compute_ll(m, template, include_H=False)
               for m in models]
        
    else:
        bins = h2stats_mod._default_bins
        models = [H2stats.from_demes(g, u) for g in args.graph_fnames]
        n_stats = models[0].stats.shape[1]
        lls = None

    conf = 0.95
    ci = stats.norm().ppf(0.5 + conf / 2)

    n_cols = 5
    n_axs = n_stats + 2

    n_rows = int(np.ceil(n_axs / n_cols))
    if n_axs < n_cols:
        n_cols = n_axs
    width = 0.125 * len(bins)
    height = width / 1.2
    fig, axs = plt.subplots(
        n_rows, 
        n_cols, 
        figsize=(n_cols * width, n_rows * height), 
        layout="constrained"
    )
    axs = axs.flat
    for ax in axs[n_axs:]:
        ax.remove()

    lines = []
    for h2stats, color in zip(datas + models, colors):
        l = plot_H2stats(
            h2stats,
            axs,
            color,
            ci,
            n_axs,
            two_sample=True
        )
        lines.append(l)
    
    if lls is not None:
        glabels = [os.path.basename(args.graph_fnames[i]) + 
                   f', ll={np.round(lls[i], 2)}'
                   for i in range(len(args.graph_fnames))]
    else:
        glabels = [os.path.basename(g) for g in args.graph_fnames]
    dlabels = [os.path.basename(d) for d in args.data_fnames]
    labels = dlabels + glabels
    y_anchor = -0.2 / n_rows
    plt.figlegend(
        lines, 
        labels, 
        framealpha=0,
        loc='lower center',
        bbox_to_anchor=(0.5, y_anchor),
        ncols=2
    )

    for i, ax in enumerate(axs):
        if ax is None:
            continue
        if i in [n_axs - 1, n_axs - 2]: 
            ax.set_ylabel('$H$')
            format_ticks(ax, x_ax=False)
        else:
            ax.set_xlabel('$r$')
            ax.set_ylabel('$H_2$')
            ax.set_xscale('log')
            ax.set_xlim(max(1e-8, 0.5 * bins[0]), 2 * bins[-1])
            format_ticks(ax)

    plt.savefig(args.out_fname, dpi=244, bbox_inches='tight')
    return 


if __name__ == "__main__":
    main()
