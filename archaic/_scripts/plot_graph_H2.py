
"""
Evaluate H2 from a demes graph and plot it alongside empirical means and CIs
"""

import argparse
import demes
import matplotlib.pyplot as plt
import numpy as np
from archaic import inference


deme_to_sample = {
    "Altai": "Altai",
    "Chagyrskaya": "Chagyrskaya",
    "Vindija": "Vindija",
    "Denisova": "Denisova",
    "Denisovan": "Denisova",
    "Yoruba": "Yoruba-1",
    "Yoruba1": "Yoruba-1",
    "Yoruba3": "Yoruba-3",
    "KhomaniSan": "Khomani_San-2",
    "Khomani_San": "Khomani_San-2",
    "Papuan": "Papuan-2",
    "French": "French-1",
    "Han": "Han-1"
}
generic_samples = list(deme_to_sample.values())
generic_demes = list(deme_to_sample.keys())


def get_args():

    parser = argparse.ArgumentParser()
    parser.add_argument("-g", "--graph_fname", required=True)
    parser.add_argument("-d", "--boot_archive", required=True)
    parser.add_argument("-o", "--out_fname", required=True)
    parser.add_argument("-n", "--name_map", type=str, nargs="*", default=[])
    parser.add_argument("-u", "--u", type=float, default=1.35e-8)
    parser.add_argument("-H", "--use_H", type=int, default=1)
    parser.add_argument("-l", "--log_scale", type=bool, default=False)
    return parser.parse_args()


def get_name_map(all_deme_names):
    # get the names that need to be loaded from data. form samples: demes
    name_map = {}
    arg_map = {x: y for x, y in [m.split(':') for m in args.name_map]}
    for deme in arg_map:
        if deme in all_deme_names:
            name_map[arg_map[deme]] = deme
        else:
            raise ValueError(f"deme {deme} is not present in the graph!")
    for deme in all_deme_names:
        if deme not in name_map.values():
            if deme in deme_to_sample:
                name_map[deme_to_sample[deme]] = deme
    if len(name_map) == 0:
        raise ValueError("sample/deme name configuration selects 0 samples!")
    sample_names = list(name_map.keys())
    deme_names = list(name_map.values())
    return name_map, sample_names, deme_names


def main():

    graph = demes.load(args.graph_fname)
    all_deme_names = [deme.name for deme in graph.demes]
    name_map, sample_names, deme_names = get_name_map(all_deme_names)
    # _graph = graph.rename_demes(reverse_map)
    r_bins, data = inference.read_data(args.boot_archive, sample_names)
    _data = inference.rename_data_samples(data, name_map)
    inference.plot(graph, _data, r_bins, log_scale=args.log_scale, use_H=args.use_H)
    plt.savefig(args.out_fname, dpi=200, bbox_inches='tight')


if __name__ == "__main__":
    args = get_args()
    main()
