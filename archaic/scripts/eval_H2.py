
"""
Usage:

"""


import argparse
import demes
import matplotlib.pyplot as plt
import numpy as np
from util import inference


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("archive_fname")
    parser.add_argument("graph_fname")
    parser.add_argument("out_fname")
    parser.add_argument("-m", "--name_map", type=str, nargs="*")
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    graph = demes.load(args.graph_fname)
    sample_dict = {x.split(":")[0]: x.split(":")[1] for x in args.name_map}
    sample_demes = list(sample_dict.keys())
    sample_names = list(sample_dict.values())
    data = inference.read_data(args.archive_fname, sample_names)
    r_bins = np.load(args.archive_fname)["r_bins"]
    _data = (sample_demes, data[1], data[2])
    inference.plot(graph, _data, r_bins)
    plt.savefig(args.out_fname, dpi=200, bbox_inches='tight')
