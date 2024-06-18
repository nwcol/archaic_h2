
"""
For a given population in a complex graph, write a one-population graph with
approximately similar coalescent rates
"""

import argparse
import demes
import numpy as np
from archaic import coalescent


def get_args():

    parser = argparse.ArgumentParser()
    parser.add_argument("-g", "--graph_fname", required=True)
    parser.add_argument("-o", "--out_fname", required=True)
    parser.add_argument("-s", "--sample_name", required=True)
    parser.add_argument("-n", "--n_epochs", type=int, default=101)
    return parser.parse_args()


def main():

    graph = demes.load(args.graph_fname)
    max_t = max([deme.start_time for deme in graph.demes[1:]])
    epoch_times = np.linspace(0, max_t, args.n_epochs)
    if graph.generation_time != 1:
        eval_times = np.concatenate(
            [epoch_times[:-1] + np.diff(epoch_times) / 2, np.array([max_t + 1])]
        ) / graph.generation_time
        out_times = epoch_times
    else:
        pass
    rates = coalescent.get_rate(graph, eval_times, args.sample_name)
    Ne = 1 / (2 * rates)
    with open(args.out_fname, 'w') as file:
        file.write(
            f"time_units: years\n"
            f"generation_time: {graph.generation_time}\n"
            f"demes:\n"
            f"  - name: {args.sample_name}\n"
            f"    epochs:\n"
        )
        for i in np.arange(args.n_epochs - 1, -1, -1):
            file.write(
                f"      - {{end_time: {out_times[i]}, start_size: {Ne[i]}}}\n"
            )
    return 0


if __name__ == "__main__":
    args = get_args()
    main()
