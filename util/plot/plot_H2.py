
import argparse
import matplotlib.pyplot as plt
import numpy as np
from util import plots
from util import file_util


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("out")
    parser.add_argument("r_bin_file")
    parser.add_argument("site_pair_file")
    parser.add_argument("het_pair_files", nargs='*')
    parser.add_argument("-y", "--y_lim", type=float, default=2e-6)
    parser.add_argument("-d", "--dpi", type=int, default=200)
    args = parser.parse_args()

    r_bins = np.loadtxt(args.r_bin_file)
    site_pairs = np.loadtxt(args.site_pair_file).sum(0)
    H2 = {}
    for file_name in args.het_pair_files:
        header, arr = file_util.load_arr(file_name)
        sample_id = header["sample_id"]
        H2[sample_id] = arr.sum(0) / site_pairs

    ax = plots.plot_r_stats(r_bins[1:], H2, markers=['x'] * len(H2))
    ax.set_ylim(0, args.y_lim)
    plt.savefig(args.out, dpi=args.dpi)
