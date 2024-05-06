
import argparse
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
from util import file_util


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("out_file_name")
    parser.add_argument("r_file_name")
    parser.add_argument("-c", "--curves", nargs="*")
    parser.add_argument("-s", "--scatters", nargs="*")
    parser.add_argument("-y", "--ylim", type=float, default=2e-6)
    parser.add_argument("-t", "--title", default=None)
    args = parser.parse_args()

    n = len(args.curves)
    colors = cm.nipy_spectral(np.linspace(0, 1, n+1))

    fig, ax = plt.subplots(figsize=(7, 6), layout="constrained")
    ax.grid(alpha=0.2)

    r = np.loadtxt(args.r_file_name)[1:]

    for i, file_name in enumerate(args.curves):
        header, vec = file_util.load_vec(file_name)
        label = f"{header['sample_id']} : moments"
        ax.plot(r, vec, color=colors[i], label=label)

    for i, file_name in enumerate(args.scatters):
        header, vec = file_util.load_vec(file_name)
        label = f"{header['sample_id']} : msprime"
        ax.scatter(r, vec, color=colors[i], label=label, marker='x')

    ax.set_xlabel("r")
    ax.set_ylabel("H2")
    ax.legend()
    ax.set_xscale("log")
    ax.set_ylim(0, args.ylim)
    plt.savefig(args.out_file_name, dpi=200)
