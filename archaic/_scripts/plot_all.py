

import argparse
import demes
import matplotlib.pyplot as plt
import numpy as np
import scipy
from archaic import inference
from archaic import plots
from archaic import utils


n_cols = 5
emp_color = "red"
E_color = "blue"


def get_args():

    parser = argparse.ArgumentParser()
    parser.add_argument("-g", "--graph_fname", required=True)
    parser.add_argument("-d", "--boot_fname", required=True)
    parser.add_argument("-o", "--out_fname", required=True)
    parser.add_argument("-u", "--u", type=float, default=1.35e-8)
    parser.add_argument("-H", "--use_H", type=int, default=1)
    parser.add_argument("-H2", "--use_H2", type=int, default=1)
    parser.add_argument("-m", "--num_method", default="simpsons")
    parser.add_argument("-a", "--alpha", type=float, default=0.05)
    parser.add_argument("-s", "--samples", nargs='*', default=[])
    return parser.parse_args()


def get_ci(cov_H, cov_H2, alpha):

    ci = scipy.stats.norm.ppf(1 - alpha / 2)
    n = len(cov_H)
    idx = np.arange(n)
    ci_H = np.sqrt(cov_H[idx, idx]) * ci
    ci_H2 = np.sqrt(cov_H2[:, idx, idx]) * ci
    return ci_H, ci_H2


def main():

    sample_names = inference.scan_names(args.graph_fname, args.boot_fname)
    pair_names = utils.get_pair_names(sample_names)
    pairs = utils.get_pairs(sample_names)
    n_samples = len(sample_names)
    n_pairs = len(pair_names)
    n_axs = 2 + n_samples + n_pairs
    n_rows = np.ceil(n_axs / n_cols).astype(int)
    ax_shape = (n_rows, n_cols)
    fig, axs = plt.subplots(
        n_rows, n_cols, figsize=(n_cols * 3, n_rows * 2), layout="constrained"
    )
    r_bins, data = inference.read_data(args.boot_fname, sample_names)
    _, __, H, cov_H, H2, cov_H2 = data
    r = inference.get_r(r_bins, method="midpoint")
    _r = inference.get_r(r_bins, method=args.num_method)
    graph = demes.load(args.graph_fname)
    E_H, E_H2 = inference.get_H_stats(
        graph, sample_names, pairs, _r, args.u, num_method=args.num_method
    )
    H_err, H2_err = get_ci(cov_H, cov_H2, args.alpha)
    plots.plot_H_err(
        axs[0, 0], H[:n_samples], H_err[:n_samples], E_H[:n_samples],
        sample_names, ['blue'] * n_samples, ["red"] * n_samples, title="$H$"
    )
    plots.plot_H_err(
        axs[0, 1], H[n_samples:], H_err[n_samples:], E_H[n_samples:],
        pair_names, ['blue'] * n_samples, ["red"] * n_samples, title="$H_{xy}$"
    )
    for i in np.arange(n_samples):
        idx = np.unravel_index(i + 2, ax_shape)
        plots.plot_H2_err(
            axs[idx], r, H2[:, i], H2_err[:, i], E_H2[:, i], emp_color,
            E_color, log_scale=True, title=f"$H_2$:{sample_names[i]}"
        )
    for i in np.arange(n_pairs):
        idx = np.unravel_index(i + 2 + n_samples, ax_shape)
        plots.plot_H2_err(
            axs[idx], r, H2[:, i], H2_err[:, i], E_H2[:, i], emp_color,
            E_color, log_scale=True, title=f"$H_2$:{pair_names[i]}"
        )
    plt.savefig(args.out_fname, dpi=200)
    return 0


if __name__ == "__main__":
    args = get_args()
    main()
