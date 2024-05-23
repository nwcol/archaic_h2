
import argparse
import demes
import matplotlib.pyplot as plt
import numpy as np
from util import inference


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("graph_fname")
    parser.add_argument("out_fname")
    parser.add_argument("-d", "--boot_archive", default=None)
    args = parser.parse_args()

    r = np.logspace(-6, -2, 41)

    graph = demes.load(args.graph_fname)
    deme_names = [x.name for x in graph.demes if x.epochs[-1].end_time == 0]
    sample_ids =

    fig, ax = plt.subplots(figsize=(7, 6), layout="constrained")

    if args.boot_archive:
        data = inference.read_data_dir(args.boot_archive, sample_ids)
        r = np.load(args.boot_archive)["r_bins"]
        inference.plot()

    else:
        pass


