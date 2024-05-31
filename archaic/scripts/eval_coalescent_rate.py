
import argparse
import demes
from matplotlib import cm
import matplotlib.pyplot as plt
import msprime
import numpy as np


linestyles = [
    "solid",
    "dashed",
    "dotted",
    "dashdot"
]


def get_args():

    parser = argparse.ArgumentParser()
    parser.add_argument("-g", "--graph_fnames", required=True, nargs='*')
    parser.add_argument("-s", "--samples", required=True, nargs='*')
    parser.add_argument("-o", "--out_fname", required=True)
    parser.add_argument("-n", "--n_samples", type=int, default=2)
    parser.add_argument("-T", "--max_T", type=int, default=5e5)
    parser.add_argument("-t", "--n_times", type=int, default=100)
    parser.add_argument("-gt", "--generation_time", type=int, default=30)
    return parser.parse_args()


def get_t():

    T = int(args.max_T / args.generation_time)
    t = np.linspace(0, T, args.n_times)
    return t


def get_rate(graph, t, samples):

    demography = msprime.Demography.from_demes(graph)
    debugger = demography.debug()
    deme_names = [deme.name for deme in graph.demes]
    rates = {}
    for sample in samples:
        if sample not in deme_names:
            raise ValueError(f"deme {sample} does not exist!")
        _rates, probs = debugger.coalescence_rate_trajectory(t, {sample: 2})
        rates[sample] = _rates
    return rates


def main():

    n = len(args.samples)
    cmap = dict(
        zip(args.samples, list(cm.gnuplot(np.linspace(0, 0.95, n))))
    )
    t = get_t()
    t_years = t * args.generation_time
    rates = {
        graph_fname: get_rate(demes.load(graph_fname), t, args.samples)
        for graph_fname in args.graph_fnames
    }
    fig, ax = plt.subplots(figsize=(8, 7), layout="constrained")
    for i, graph_name in enumerate(rates):
        linestyle = linestyles[i]
        rate_dict = rates[graph_name]
        for j, sample in enumerate(rate_dict):
            ax.plot(
                t_years, rate_dict[sample], label=f"{graph_name}: {sample}",
                color=cmap[sample], linestyle=linestyle
            )
    ax.legend()
    ax.set_xlim(0, )
    ax.set_ylim(0, )
    ax.set_xlabel("time, years")
    ax.set_ylabel("coalescent rate")
    ax.grid(alpha=0.2)
    plt.savefig(args.out_fname, dpi=200)


if __name__ == "__main__":
    args = get_args()
    main()
