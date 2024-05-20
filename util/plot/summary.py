"""
Make quick summary plots of H2 from .npz archives
"""

import argparse
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("archive_name")
    parser.add_argument("out_prefix")
    parser.add_argument("-y1", "--y_max1", type=float, default=2e-6)
    parser.add_argument("-y2", "--y_max2", type=float, default=2e-7)
    args = parser.parse_args()

    genome_stats = np.load(args.archive_name)
    fields = genome_stats.files
    sample_ids = genome_stats["sample_ids"]
    sample_pairs = genome_stats["sample_pairs"]

    r_bins = genome_stats["r_bins"]
    r = r_bins[1:]

    colors = cm.nipy_spectral(np.linspace(0, 0.9, len(sample_ids)))

    # H plot?

    # main H2 plot
    H2 = genome_stats["H2"]
    std_H2 = np.std(H2, axis=1)
    fig, ax = plt.subplots(figsize=(7, 6), layout="constrained")
    ax.grid(alpha=0.2)
    ax.set_xscale("log")
    for i, sample_id in enumerate(sample_ids):
        ax.plot(r, H2[i], label=sample_id, color=colors[i])
    ax.set_xlabel("r bin")
    ax.set_ylabel("H2")
    ax.legend(fontsize=9)
    ax.set_ylim(0, args.y_max1)
    plt.savefig(f"{args.out_prefix}_H2.png", dpi=200)
    ax.set_ylim(0, args.y_max2)
    plt.savefig(f"{args.out_prefix}_H2_archaic.png", dpi=200)
    plt.close()

    # cross-sample H2 plots

