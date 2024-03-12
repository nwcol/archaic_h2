
import argparse
import json
import matplotlib.pyplot as plt
import numpy as np
from util import file_util
from util import plot_fxns


def get_x_vec(window_dicts):

    chrom_numbers = list(window_dicts.keys())
    chrom_numbers.sort()
    x_vec = []
    chr_vec = []
    offset = 0
    for i in chrom_numbers:
        window_dict = window_dicts[i]
        window_ids = list(window_dict.keys())
        upper_bounds = [
            window_dict[x]["bounds"][1] + offset for x in window_ids
        ]
        chr_vec += [i] * len(upper_bounds)
        offset = max(upper_bounds)
        x_vec += upper_bounds
    x_vec = np.array(x_vec)
    chr_vec = np.array(chr_vec)
    return x_vec, chr_vec


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("stat_file_name")
    parser.add_argument("out_file_name")
    parser.add_argument("-c", "--column_idx", type=int, default=0)
    parser.add_argument("-y", "--y_lim", type=float, default=2e-4)
    args = parser.parse_args()
    #
    header, arr = file_util.load_arr(args.stat_file_name)
    window_dict = header["windows"]
    x_vec, chr_vec = get_x_vec(window_dict)
    col_idx = args.column_idx
    col_name = header["cols"][col_idx]
    sample_id = header["sample_id"]
    title = f"bin {col_idx}, r_{col_name}, {sample_id}"
    plot_fxns.manhattan_plot(
        arr[:, col_idx], x_vec, chr_vec, args.y_lim, title
    )
    plt.savefig(args.out_file_name, dpi=200)
