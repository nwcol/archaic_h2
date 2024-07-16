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
    parser.add_argument("-g", "--real_graph_fname", required=True)
    parser.add_argument("-g1", "--graph_fnames1", nargs='*', required=True)
    parser.add_argument("-g2", "--graph_fnames2", nargs='*', required=True)
    parser.add_argument("-label1", '--label1', default=None)
    parser.add_argument("-label2", '--label2', default=None)
    parser.add_argument("-px", "--param_x", required=True)
    parser.add_argument("-py", "--param_y", required=True)
    parser.add_argument("-o", "--out_fname", required=True)
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
    idx_x = names.index(args.param_x)
    idx_y = names.index(args.param_y)
    fig, ax = plt.subplots(figsize=(5.5, 5), layout="constrained")
    if args.label1:
        label = args.label1
    else:
        label = args.graph_fnames1[0].split('/')[0]
    ax.scatter(params1[:, idx_x], params1[:, idx_y], color=color1, marker='x',
               label=label)
    if args.label2:
        label = args.label2
    else:
        label = args.graph_fnames2[0].split('/')[0]
    ax.scatter(params2[:, idx_x], params2[:, idx_y], color=color2, marker='x',
               label=label)
    ax.set_xlim(lower[idx_x], upper[idx_x])
    ax.set_ylim(lower[idx_y], upper[idx_y])
    ax.set_xlabel(args.param_x)
    ax.set_ylabel(args.param_y)
    ax.grid(alpha=0.2)
    ax.scatter(init[idx_x], init[idx_y], color="black", label="true", marker='x')
    ax.legend()
    plt.savefig(args.out_fname, dpi=200)


def main():

    args = get_args()
    plot(args)
    return 0


if __name__ == "__main__":
    main()
