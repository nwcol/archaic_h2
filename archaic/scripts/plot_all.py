
"""
Plot an arbitrary number of H, H2 expectations alongside 0 or 1 empirical vals.
"""

import argparse
import demes
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import scipy
from archaic import inference
from archaic import plots
from archaic import utils


def get_args():
    # get args
    parser = argparse.ArgumentParser()
    parser.add_argument("-g", "--graph_fnames", nargs='*', default=[])
    parser.add_argument("-a", "--archive_fnames", nargs='*', default=[])
    parser.add_argument("-d", "--boot_fnames", nargs='*', default=[])
    parser.add_argument("-o", "--out_fname", required=True)
    parser.add_argument("-u", "--u", type=float, default=1.35e-8)
    parser.add_argument("-H", "--use_H", type=int, default=1)
    parser.add_argument("-H2", "--use_H2", type=int, default=1)
    parser.add_argument("-m", "--num_method", default="simpsons")
    parser.add_argument("-alpha", "--alpha", type=float, default=0.05)
    parser.add_argument("-s", "--sample_names", nargs='*', default=[])
    parser.add_argument("-ly", "--log_y", type=int, default=0)
    parser.add_argument("-yH", "--H_ylim", type=float, default=None)
    parser.add_argument("-yH2", "--H2_ylim", type=float, default=None)
    parser.add_argument("-neg", "--allow_negatives", type=int, default=0)
    parser.add_argument("-t", "--fig_title", default=None)
    parser.add_argument("--n_cols", type=int, default=5)
    return parser.parse_args()


def get_ci(cov_H, cov_H2, alpha):
    # get error bars
    ci = scipy.stats.norm.ppf(1 - alpha / 2)
    n = len(cov_H)
    idx = np.arange(n)
    ci_H = np.sqrt(cov_H[idx, idx]) * ci
    ci_H2 = np.sqrt(cov_H2[:, idx, idx]) * ci
    return ci_H, ci_H2


def main():
    #
    if len(args.sample_names) > 0:
        sample_names = args.sample_names
    else:
        if len(args.graph_fnames) > 0:
            sample_names = inference.scan_names(
                args.graph_fnames[0], args.boot_fnames[0]
            )
            # guess sample names from the graph
        else:
            if len(args.boot_fnames) > 0:
                sample_names = list(np.load(args.boot_fnames[0])["sample_names"])
    pairs = utils.get_pairs(sample_names)
    pair_names = utils.get_pair_names(sample_names)
    abbrev_names = [name[:3] for name in sample_names]
    abbrev_pairs = [f"{x[:3]}-{y[:3]}" for x, y in pairs]
    n_samples = len(sample_names)
    n_pairs = len(pair_names)
    if n_pairs > 0:
        offset = 2
    else:
        offset = 1
    n_plots = offset + n_samples + n_pairs
    if n_plots < args.n_cols:
        n_rows = 1
        n_cols = n_plots
    else:
        n_rows = np.ceil(n_plots / args.n_cols).astype(int)
        n_cols = args.n_cols
    shape = (n_rows, n_cols)
    fig, axs = plt.subplots(
        n_rows, n_cols, figsize=(n_cols * 3.5, n_rows * 2.5), layout="constrained"
    )
    if n_rows == 1:
        axs = axs[np.newaxis, :]
    if len(args.boot_fnames) > 0:
        r_bins = np.load(args.boot_fnames[0])["r_bins"]
    else:
        r_bins = np.logspace(-6, -2, 30)
    r = inference.get_r(r_bins, method="midpoint")
    _r = inference.get_r(r_bins, method=args.num_method)
    n_boots = len(args.boot_fnames)
    n_graphs = len(args.graph_fnames)
    n_archives = len(args.archive_fnames)
    b_colors = plots.get_terrain_cmap(n_boots)
    colors = plots.get_gnu_cmap(n_graphs + n_archives)
    for k, fname in enumerate(args.boot_fnames):
        r_bins, data = inference.read_data(fname, sample_names)
        __r = inference.get_r(r_bins, method="midpoint")
        _, __, H, cov_H, H2, cov_H2 = data
        H_err, H2_err = get_ci(cov_H, cov_H2, args.alpha)
        label = fname.replace(".npz", "")
        plots.plot_error_points(
            axs, H, H_err, H2, H2_err, r, abbrev_names, abbrev_pairs,
            b_colors[k], args.log_y, label=label
        )
    for k, fname in enumerate(args.graph_fnames):
        graph = demes.load(fname)
        E_H, E_H2 = inference.get_H_stats(
            graph, sample_names, pairs, _r, args.u, num_method=args.num_method
        )
        label = fname.replace(".yaml", "")
        if len(args.boot_fnames) > 0:
            r_bins, data = inference.read_data(args.boot_fnames[0], sample_names)
            samples, pairs, H, H_cov, H2, H2_cov = data
            data = (samples, pairs, H, np.linalg.inv(H_cov), H2, np.linalg.inv(H2_cov))
            lik = inference.eval_log_lik(graph, data, _r, args.u)
            print(lik)
            label += f": LL = {np.round(lik, 0)}"
        plots.plot_curves(
            axs, E_H, E_H2, r, abbrev_names, abbrev_pairs, colors[k],
            args.log_y, label=label
        )
    for k, fname in enumerate(args.archive_fnames):
        archive = np.load(fname)
        __r = archive["r"]
        H = archive["H"]
        H2 = archive["H2"]
        label = fname.replace(".npz", "")
        plots.plot_curves(
            axs, H, H2, __r, abbrev_names, abbrev_pairs, colors[k + n_graphs],
            args.log_y, label=label
        )
    if args.allow_negatives:
        bottom = None
    else:
        bottom = 0
    if n_pairs == 0:
        offset = 1
    else:
        offset = 2
    for i in np.arange(0, offset):
        idx = np.unravel_index(i, shape)
        if args.H_ylim:
            axs[idx].set_ylim(bottom, args.H_ylim)
        else:
            axs[idx].set_ylim(bottom=bottom)
    for i in np.arange(offset, n_plots):
        idx = np.unravel_index(i, shape)
        if not args.log_y:
            if args.H2_ylim:
                axs[idx].set_ylim(bottom, args.H2_ylim)
            else:
                axs[idx].set_ylim(bottom=bottom)
    fig.legend(
        loc="lower center", shadow=True, fontsize=9, ncols=4,
        bbox_to_anchor=(0.5, -0.1)
    )
    if args.fig_title:
        fig.title(args.fig_title)
    axs = axs.flat
    for ax in axs[2 + n_samples + n_pairs:]:
        ax.remove()
    plt.savefig(args.out_fname, dpi=200, bbox_inches='tight')
    return 0


if __name__ == "__main__":
    args = get_args()
    main()
