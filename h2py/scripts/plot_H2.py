"""
Plot an arbitrary number of H, H2 expectations alongside 0 or 1 empirical vals.
"""
import argparse
import demes
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
from scipy import stats

from h2py import inference
from h2py.h2stats_mod import H2stats


def get_args():
    # get args
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--data_fname', required=True)
    parser.add_argument('-g', '--graph_fname', required=True)
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
                axs[k].errorbar(
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
                axs[k].plot(x, H2, color=color)
                axs[Hax].scatter(hpos, H, color=color, marker='x')

            axs[k].set_title(sample_i)
            k += 1
    axs[n_axs - 2].set_xticks(np.arange(len(hlabels)), labels=hlabels, rotation=90)
    axs[n_axs - 1].set_xticks(np.arange(len(hxylabels)), labels=hxylabels, rotation=90)
    return


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
    if args.u is not None:
        u = args.u
    else:
        u = demes.load(args.graph_fname).metadata['opt_info']['u']
    data = H2stats.from_file(args.data_fname, graph=args.graph_fname)
    model = H2stats.from_demes(args.graph_fname, u, template=data)
    bins = data.bins
    ll = inference.compute_ll(model, data, include_H=False)
    
    conf = 0.95
    ci = stats.norm().ppf(0.5 + conf / 2)

    n_cols = 5
    n_axs = data.stats.shape[1] + 2

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

    for h2stats, color in zip([model, data], ['blue', 'green']):
        plot_H2stats(
            h2stats,
            axs,
            color,
            ci,
            n_axs,
            two_sample=True
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
