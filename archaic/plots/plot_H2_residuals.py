"""
Currently: residuals are normalized
"""


import argparse
import demes
import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
from archaic import inference, plotting, utils
from archaic.spectra import H2Spectrum


def get_args():
    # get args
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--data_fname', required=True)
    parser.add_argument('-g', '--graph_fnames', nargs='*', required=True)
    parser.add_argument('-o', '--out_fname', required=True)
    parser.add_argument('-u', '--u', type=float, default=1.35e-8)
    parser.add_argument('--sample_ids', nargs='*', default=[])
    parser.add_argument('--fig_title', default=None)
    parser.add_argument('--n_cols', type=int, default=5)
    parser.add_argument('--uniform_ylim', type=int, default=1)
    return parser.parse_args()


def main():
    args = get_args()

    if args.sample_ids:
        sample_ids = args.sample_ids
        graph = None
    else:
        sample_ids = None
        graph = demes.load(args.graph_fnames[0])

    data = H2Spectrum.from_bootstrap_file(
        args.data_fname,
        sample_ids=sample_ids,
        graph=demes.load(args.graph_fnames[0])
    )

    # used to compute graph H2 values
    _sample_ids = data.sample_ids

    # getting r
    r_bins = data.r_bins
    r = H2Spectrum.get_r(r_bins)

    labels = []
    differences = []

    for graph_fname in args.graph_fnames:
        graph = demes.load(graph_fname)
        model = H2Spectrum.from_graph(
            graph, _sample_ids, r, args.u, r_bins=r_bins
        )
        ll = inference.get_ll(model, data)
        ll_label = f', ll={np.round(ll, 0)}'
        difference = H2Spectrum(
            (model.data - data.data) / data.data,
            r_bins,
            model.ids,
            sample_ids=sample_ids,
            has_H=data.has_H,
            r=r
        )
        differences.append(difference)
        basename = graph_fname.split('/')[-1]
        labels.append(f'{basename}{ll_label}')

    colors = list(cm.gnuplot(np.linspace(0, 0.9, len(differences))))

    fig, axs = plotting.plot_H2_spectra(
        *differences,
        plot_H=True,
        colors=colors,
        labels=labels,
        n_cols=args.n_cols,
        ylim_0=False
    )

    if args.uniform_ylim:
        max_res = max([np.max(np.abs(diffs.data)) for diffs in differences])
        for ax in axs:
            ax.set_ylim(-max_res, max_res)

    if args.fig_title:
        fig.title(args.fig_title)
    plt.savefig(args.out_fname, dpi=200, bbox_inches='tight')
    return 0


if __name__ == "__main__":
    main()