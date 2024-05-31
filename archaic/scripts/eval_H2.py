
import argparse
import demes
import matplotlib.pyplot as plt
import numpy as np
from archaic import inference


def get_args():

    parser = argparse.ArgumentParser()
    parser.add_argument("-g", "--graph_fname", required=True)
    parser.add_argument("-d", "--boot_archive", required=True)
    parser.add_argument("-o", "--out_fname", required=True)
    parser.add_argument("-n", "--name_map", type=str, nargs="*", required=True)
    parser.add_argument("-u", "--u", type=float, default=1.35e-8)
    parser.add_argument("-H", "--use_H", type=int, default=1)
    parser.add_argument("-l", "--log_scale", type=bool, default=False)
    return parser.parse_args()


def map_names(name_map_strs):
    # of the form deme_name:archive_sample_name
    name_map = {}
    for mapping in name_map_strs:
        x, y = mapping.split(":")
        name_map[x] = y
    deme_names = list(name_map.keys())
    sample_names = list(name_map.values())
    print(name_map)
    return name_map, deme_names, sample_names


def main():

    graph = demes.load(args.graph_fname)
    name_map, deme_names, sample_names = map_names(args.name_map)
    reverse_map = {
        x: name_map[x].replace("-", "") for x in name_map
    }
    # _graph = graph.rename_demes(reverse_map)
    data = inference.read_data(args.boot_archive, sample_names)
    _data = (deme_names, data[1], data[2])
    r_bins = np.load(args.boot_archive)["r_bins"]
    inference.plot(graph, _data, r_bins, log_scale=args.log_scale, use_H=args.use_H)
    plt.savefig(args.out_fname, dpi=200, bbox_inches='tight')


if __name__ == "__main__":
    args = get_args()
    main()
