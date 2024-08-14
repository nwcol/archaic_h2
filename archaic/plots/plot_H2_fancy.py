"""
Plot an arbitrary number of H, H2 expectations alongside 0 or 1 empirical vals.
"""
import argparse
import demes
import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np

from archaic import inference, plotting
from archaic.spectra import H2Spectrum


def get_args():
    # get args
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--data_fname', required=True)
    parser.add_argument('-g', '--graph_fname', required=True)
    parser.add_argument("-o", "--out_fname", required=True)
    parser.add_argument("-u", "--u", type=float, default=1.35e-8)
    parser.add_argument('--alpha', type=float, default=0.05)
    parser.add_argument('--labels', nargs='*', default=None)
    return parser.parse_args()


def main():

    args = get_args()
    graph = demes.load(args.graph_fname)
    data = H2Spectrum.from_bootstrap_file(args.data_fname, graph=graph)
    model = H2Spectrum.from_graph(graph, data.sample_ids, data.r, args.u)

    if data.arr.shape[1] < 8:
        colors = ['b', 'orange', 'g', 'r', 'purple', 'brown', 'm', 'g']
    else:
        colors = list(cm.gnuplot(np.linspace(0, 0.9, data.arr.shape[1])))
    #
    plotting.plot_two_panel_H2(
        model,
        data,
        args.labels,
        colors,
        args.out_fname,
    )
    return 0


if __name__ == "__main__":
    main()
