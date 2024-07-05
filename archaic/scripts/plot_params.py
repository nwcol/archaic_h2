"""
Plots parameter distributions and writes variances
"""


import argparse
import matplotlib.pyplot as plt
import moments.Demes.Inference as minf
import numpy as np
from archaic import inference
from archaic import plotting


def get_args():
    # get args
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--params_fname", required=True)
    parser.add_argument("-g", "--real_graph_fname", required=True)
    parser.add_argument("-g1", "--graph_fnames1", nargs='*', required=True)
    parser.add_argument("-g2", "--graph_fnames2", nargs='*', required=True)
    parser.add_argument("-o", "--out_fname", required=True)
    parser.add_argument("-b", "--n_bins", type=int, default=50)
    return parser.parse_args()


def parse_params(graph_fname, params_fname):
    # grab
    builder = minf._get_demes_dict(graph_fname)
    pars = minf._get_params_dict(params_fname)
    names, init, lower, upper = minf._set_up_params_and_bounds(pars, builder)
    return names, init, lower, upper


def plot(args):
    #
    color1 = "blue"
    color2 = "red"
    names, init, lower, upper = parse_params(
        args.real_graph_fname, args.params_fname
    )
    _names, params1 = inference.parse_graph_params(
        args.params_fname, args.graph_fnames1
    )
    _names, params2 = inference.parse_graph_params(
        args.params_fname, args.graph_fnames2
    )
    n_params = len(names)
    n_cols = min(n_params, 3)
    n_rows = int(np.ceil(n_params / n_cols))
    fix, axs = plt.subplots(
        n_rows, n_cols, figsize=(n_cols * 4, n_rows * 3), layout="constrained"
    )
    axs = axs.flat
    for i, param in enumerate(names):
        bins = np.linspace(lower[i], upper[i], args.n_bins)
        plotting.plot_distribution(
            bins,
            params1[:, i],
            axs[i],
            color=color1,
            label=args.graph_fnames1[0].split('/')[0]
        )
        plotting.plot_distribution(
            bins,
            params2[:, i],
            axs[i],
            color=color2,
            label=args.graph_fnames2[0].split('/')[0]
        )
        axs[i].set_xlabel(param)
        axs[i].set_ylim(0, )
        axs[i].scatter(init[i], 0, marker='x', color="black")
    for ax in axs[n_params:]:
        ax.remove()
    plt.savefig(args.out_fname, dpi=200)


def main():

    args = get_args()
    plot(args)
    return 0


if __name__ == "__main__":
    main()
