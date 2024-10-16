"""
Plot an arbitrary number of H, H2 expectations alongside 0 or 1 empirical vals.
"""
import argparse
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib import cm
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
    two_sample=True
):

    x = h2stats.bins[:-1] + np.diff(h2stats.bins) / 2
    k = 0
    for i, sample_i in enumerate(h2stats.pop_ids):
        for sample_j in h2stats.pop_ids[i:]:
            if sample_i == sample_j: 
                stat = h2stats.stats[:-1, k]
                if h2stats.covs is not None:
                    std = h2stats.covs[:-1, k, k] ** 0.5 * ci
                    print(std)
                    axs[k].errorbar(
                        x, 
                        stat, 
                        yerr=std, 
                        markerfacecolor='none',
                        markeredgecolor=color, 
                        ecolor=color,
                        fmt="o", 
                        capsize=0
                    )
                else:
                    axs[k].plot(x, stat, color=color)
                axs[k].set_title(sample_i)

            else:
                if not two_sample: 
                    continue
                stat = h2stats.stats[:-1, k]
                if h2stats.covs is not None:
                    std = h2stats.covs[:-1, k, k] ** 0.5 * ci
                    axs[k].errorbar(
                        x, 
                        stat, 
                        yerr=std, 
                        markerfacecolor='none',
                        markeredgecolor=color, 
                        ecolor=color,
                        fmt="o", 
                        capsize=0
                    )
                else:
                    axs[k].plot(x, stat, color=color)
                axs[k].set_title(f'{sample_i[:3]},{sample_j[:3]}')
            
            k += 1
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
    data = H2stats.from_file(args.data_fname, graph=args.graph_fname)
    model = H2stats.from_demes(args.graph_fname, args.u, template=data)
    bins = data.bins
    ll = inference.compute_ll(model, data, include_H=False)
    
    conf = 0.975
    ci = stats.norm().ppf(0.95)
    print(ci)

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

    for h2stats, color in zip([model, data], ['blue', 'green']):
        plot_H2stats(
            h2stats,
            axs,
            color,
            ci,
            two_sample=True
        )

    for ax in axs[n_axs:]:
        ax.remove()

    for i, ax in enumerate(axs):
        if ax is None:
            continue
        ax.set_xscale('log')
        ax.set_xlim(max(1e-8, 0.5 * bins[0]), 2 * bins[-1])
        ax.set_xlabel('$r$')
        ax.set_ylabel('$H_2$')
        format_ticks(ax)

    plt.savefig(args.out_fname, dpi=244, bbox_inches='tight')
    return 


if __name__ == "__main__":
    main()
