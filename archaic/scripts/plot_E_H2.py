
import argparse
import demes
import matplotlib.pyplot as plt
from matplotlib import cm
import moments
import numpy as np
from archaic import plots
from archaic import utils


def get_args():

    parser = argparse.ArgumentParser()
    parser.add_argument("-g", "--graph_fnames", nargs="*", required=True)
    parser.add_argument("-s", "--samples", type=str, nargs="*", required=True)
    parser.add_argument("-o", "--out_fname", required=True)
    parser.add_argument("-u", "--u", type=float, default=1.35e-8)
    parser.add_argument("-y", "--y_lim", type=float, default=None)
    return parser.parse_args()


def get_moments_stats(graph, samples, pairs, r, u):

    ld_stats = moments.LD.LDstats.from_demes(
        graph,
        sampled_demes=samples,
        theta=None,
        r=r,
        u=u
    )
    H2 = np.array(
        [ld_stats.H2(sample, phased=True) for sample in samples] +
        [ld_stats.H2(id_x, id_y, phased=False) for id_x, id_y in pairs]
    )
    n = len(samples)
    H = np.array(
        [ld_stats.H(pops=[i])[0] for i in range(n)] +
        [ld_stats.H(pops=pair)[1] for pair in utils.get_pair_idxs(n)]
    )
    return H2, H


def main():

    samples = args.samples
    pairs = utils.get_pairs(samples)
    n = len(samples) + len(pairs)
    colors = list(cm.gnuplot(np.linspace(0, 0.95, n)))
    r = np.logspace(-6, -2, 100)
    fig, ax = plt.subplots(figsize=(8, 7), layout="constrained")
    for i, graph_name in enumerate(args.graph_fnames):
        graph = demes.load(graph_name)
        abbrev_graph_name = graph_name.replace(".yaml", "")
        H2, H = get_moments_stats(graph, samples, pairs, r, args.u)
        _names = samples + [f"{x}-{y}" for x, y in pairs]
        names = [f"{abbrev_graph_name}: {_name}" for _name in _names]
        plots.plot_H2(
            ax, r, H2, names, colors, styles=plots.line_styles[i]
        )
    if args.y_lim:
        ax.set_ylim(0, args.y_lim)
    ax.legend()
    plt.savefig(args.out_fname, dpi=200)


if __name__ == "__main__":
    args = get_args()
    main()
