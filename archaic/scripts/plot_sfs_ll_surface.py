
"""
Params file must contain two unconstrained parameters with lower, upper bound
"""

import argparse
import demes
import matplotlib
import matplotlib.pyplot as plt
import moments.Demes.Inference as minf
import moments
import numpy as np
from archaic import inference


def get_args():
    # get args
    parser = argparse.ArgumentParser()
    parser.add_argument("-g", "--graph_fname", required=True)
    parser.add_argument("-p", "--params_fname", required=True)
    parser.add_argument("-d", "--data_fname", required=True)
    parser.add_argument("-o", "--out_fname", required=True)
    parser.add_argument("-n", "--n_points", type=int, default=25)
    parser.add_argument("-l", "--n_levels", type=int, default=50)
    parser.add_argument("-u", "--u", type=float, default=1.35e-8)
    parser.add_argument("-L", "--L", type=float, required=True)
    return parser.parse_args()


def get_ll(graph, data, deme1, deme2, u, L):

    sfs = moments.Demes.SFS(graph, samples={deme1: 2, deme2: 2}, u=u)
    # get total SFS over genome
    sfs *= L
    # Poisson likelihood
    ll = moments.Inference.ll(sfs, data)
    return ll


def update_graph(builder, options, params):

    builder = minf._update_builder(builder, options, params)
    graph = demes.Graph.fromdict(builder)
    return graph


def main():

    builder = minf._get_demes_dict(args.graph_fname)
    options = minf._get_params_dict(args.params_fname)
    param_names, params_0, lower_bounds, upper_bounds = \
        minf._set_up_params_and_bounds(options, builder)
    xlabel, ylabel = param_names
    n = args.n_points
    x_bounds = np.linspace(lower_bounds[0], upper_bounds[0], n + 1)
    y_bounds = np.linspace(lower_bounds[1], upper_bounds[1], n + 1)
    x = x_bounds[:-1] + (x_bounds[1:] - x_bounds[:-1]) / 2
    y = y_bounds[:-1] + (y_bounds[1:] - y_bounds[:-1]) / 2
    sample_names = inference.scan_names(args.graph_fname, args.data_fname)
    deme1 = sample_names[0]
    deme2 = sample_names[1]
    sfs = np.load(args.data_fname)["sfs_arr"]
    data = moments.Spectrum(sfs)
    Z = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            graph = update_graph(builder, options, [x[i], y[j]])
            Z[j, i] = get_ll(graph, data, deme1, deme2, args.u, args.L)
            print(x[i], y[j], Z[j, i])
    levels = - np.logspace(np.log10(-Z.min()), np.log10(-Z.max()), args.n_levels)
    fig, ax = plt.subplots(figsize=(9, 7), layout="constrained")
    cmap = plt.colormaps['PiYG']
    norm = matplotlib.colors.BoundaryNorm(levels, ncolors=cmap.N)
    im = ax.pcolormesh(x, y, Z, cmap=cmap, norm=norm)
    fig.colorbar(im, ax=ax)
    x0, y0 = params_0
    ax.scatter(x0, y0, marker='x', color="black")
    ax.set_ylabel(ylabel)
    ax.set_xlabel(xlabel)
    plt.savefig(args.out_fname, dpi=200)
    return 0


if __name__ == "__main__":
    args = get_args()
    main()
