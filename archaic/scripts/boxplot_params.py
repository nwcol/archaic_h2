"""

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
    parser.add_argument("-g", "--graph_fname", required=True)
    parser.add_argument("-g1", "--g1", nargs='*', default=[])
    parser.add_argument("-g2", "--g2", nargs='*', default=[])
    parser.add_argument("-g3", "--g3", nargs='*', default=[])
    parser.add_argument('--labels', nargs='*', default=[])
    parser.add_argument('--title', default=None)
    parser.add_argument('--marker_size', type=int, default=2)
    parser.add_argument("-o", "--out_fname", required=True)
    return parser.parse_args()


def parse_params(graph_fname, params_fname):
    # grab
    builder = minf._get_demes_dict(graph_fname)
    pars = minf._get_params_dict(params_fname)
    names, init, lower, upper = minf._set_up_params_and_bounds(pars, builder)
    return names, init, lower, upper


def main():
    #
    args = get_args()
    param_names, real_vals, lower, upper = parse_params(
        args.graph_fname, args.params_fname
    )
    bounds = np.array([lower, upper]).T
    arrs = []
    for fnames in [args.g1, args.g2, args.g3]:
        if len(fnames) > 0:
            _, arr = inference.parse_graph_params(
                args.params_fname, fnames, permissive=True
            )
            arrs.append(arr)
    labels = args.labels
    if len(labels) != len(arrs):
        raise ValueError('label length mismatches graph category length')
    plotting.box_plot_parameters(
        param_names,
        real_vals,
        bounds,
        labels,
        *arrs,
        title=args.title
    )
    plt.savefig(args.out_fname, dpi=200, bbox_inches='tight')


if __name__ == '__main__':
    main()
