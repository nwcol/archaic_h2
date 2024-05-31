
import argparse
import demes
import demesdraw
import matplotlib.pyplot as plt
from matplotlib import cm
import moments
import numpy as np
from archaic import one_locus


def get_args():

    parser = argparse.ArgumentParser()
    parser.add_argument("-g", "--graph_fname", required=True)
    parser.add_argument("-s", "--samples", type=str, nargs="*", required=True)
    parser.add_argument("-o", "--out_fname", required=True)
    parser.add_argument("-u", "--u", type=float, default=1.35e-8)
    parser.add_argument("-y", "--ylim", type=float, default=3e-6)
    parser.add_argument("-m", "--max_time", type=int, default=1_000_000)
    return parser.parse_args()


def get_moments_stats(graph, samples, sample_pairs, r, u):

    ld_stats = moments.LD.LDstats.from_demes(
        graph,
        sampled_demes=samples,
        theta=None,
        r=r,
        u=u
    )
    H2 = np.array(
        [ld_stats.H2(sample, phased=True) for sample in samples] +
        [ld_stats.H2(id_x, id_y, phased=False) for id_x, id_y in sample_pairs]
    ).T
    n = len(samples)
    H = np.array(
        [ld_stats.H(pops=[i])[0] for i in range(n)] +
        [ld_stats.H(pops=pair)[1] for pair in one_locus.enumerate_indices(n)]
    )
    return H2, H


def plot_graph(ax, graph, color_map):

    demesdraw.tubes(graph, ax=ax, colours=color_map, max_time=args.max_time)


def plot_H(ax, H, names, colors):

    abbrev_names = []
    for i, name in enumerate(names):
        if type(name) == str:
            name = name[:3]
        else:
            name = f"{name[0][:3]}-{name[1][:3]}"
        abbrev_names.append(name)
        ax.scatter(i, H[i], color=colors[i], marker='.')
    ax.set_ylim(0, )
    ax.set_ylabel("$H$")
    ax.set_xticks(np.arange(len(names)), abbrev_names)
    ax.grid(alpha=0.2)


def plot_H2(ax, r, H2, names, colors):

    for i, name in enumerate(names):
        if type(name) == str:
            style = "solid"
        else:
            style = "dotted"
            name = f"{name[0]}-{name[1]}"
        ax.plot(r, H2[:, i], color=colors[i], linestyle=style, label=name)
    ax.set_xscale("log")
    ax.set_ylabel("$H_2$")
    ax.set_xlabel("r")
    ax.grid(alpha=0.2)
    ax.set_ylim(0, args.ylim)
    return ax


def main():

    samples = args.samples
    sample_pairs = one_locus.enumerate_pairs(samples)
    r = np.logspace(-6, -1, 30)
    graph = demes.load(args.graph_fname)
    H2, H = get_moments_stats(graph, samples, sample_pairs, r, args.u)
    fig, axs = plt.subplot_mosaic(
        [[0, 1], [2, 2]], figsize=(10, 8), layout="constrained"
    )
    sample_colors = list(cm.gnuplot(np.linspace(0, 0.95, len(samples))))
    pair_colors = list(cm.terrain(np.linspace(0, 0.90, len(sample_pairs))))
    names = samples + sample_pairs
    colors = sample_colors + pair_colors
    color_map = {sample: sample_colors[i] for i, sample in enumerate(samples)}
    plot_graph(axs[0], graph, color_map)
    plot_H(axs[1], H, names, colors)
    plot_H2(axs[2], r, H2, names, colors)
    fig.legend(fontsize=9, loc='center left', bbox_to_anchor=(1, 0.5))
    plt.savefig(args.out_fname, dpi=200, bbox_inches='tight')


if __name__ == "__main__":
    args = get_args()
    main()
