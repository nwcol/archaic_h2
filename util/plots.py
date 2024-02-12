
#

import matplotlib
import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
import sys
import os
from util import two_locus
from util.two_locus import r
from util import bed_util


data_path = "/home/nick/Projects/archaic/data"
stat_path = "/home/nick/Projects/archaic/statistics"
pair_counts = np.loadtxt(f"{stat_path}/two_locus/all/pair_counts_all.txt")


if __name__ == "__main__":
    plt.rcParams['figure.dpi'] = 100
    matplotlib.use('Qt5Agg')


def correct(paths):

    def save_arr(arr, header, file_name):
        file = open(file_name, "w")
        np.savetxt(file, arr, header=header)
        file.close()

    for path in paths:
        outname = f"{stat_path}/{path.split('/')[-1]}"
        header, arr = load_arr(path)
        header["rows"] = {0: "pair_counts"}
        save_arr(arr, str(header), outname)


def load_arr(path):

    with open(path, 'r') as file:
        header_line = file.readline().strip("\n").strip("#")
    header = eval(header_line)
    if type(header["rows"]) == str:
        header["rows"] = eval(header["rows"])
    arr = np.loadtxt(path)
    if arr.ndim < 2:
        arr = arr[np.newaxis, :]
    return header, arr


def load_statistics(path):

    header, arr = load_arr(path)
    rows = header["rows"]
    out = {
        "statistic": header["statistic"],
        "chrom": header["chrom"],
        "window_id": header["window_id"],
        "rows": {rows[i]: arr[i] for i in rows}
    }
    return out


def get_statistic_arr(stat_dicts):

    n_rows = len(stat_dicts)
    sample_ids = list(stat_dicts[0]["rows"].keys())
    n_cols = len(stat_dicts[0]["rows"][sample_ids[0]])
    arr_dict = {sample_id: np.zeros((n_rows, n_cols))
                for sample_id in sample_ids}

    all_dicts = {(int(x["chrom"]), int(x["window_id"])): x["rows"]
                 for x in stat_dicts}
    dict_ids = list(all_dicts.keys())
    dict_ids.sort()
    idx = 0

    for ids in dict_ids:
        window_dict = all_dicts[ids]

        for sample_id in sample_ids:
            arr_dict[sample_id][idx] = window_dict[sample_id]
        idx += 1

    return arr_dict


def bootstrap_two_locus(pair_counts, arr, sample_size, n_resamplings):
    """
    Bootstrap over an array of window statistics

    :param pair_counts:
    :param arr: arr of stats
    :param sample_size:
    :param n_resamplings:
    :return:
    """
    n_rows, n_cols = np.shape(arr)
    out_arr = np.zeros((n_resamplings, n_cols))

    for i in np.arange(n_resamplings):
        sample_idx = np.random.choice(np.arange(n_rows), size=sample_size,
                                      replace=False)
        sum_statistic = np.sum(arr[sample_idx], axis=0)
        sum_pair_counts = np.sum(pair_counts[sample_idx], axis=0)
        bootstrap_stat = sum_statistic / sum_pair_counts
        out_arr[i] = bootstrap_stat

    out_arr = np.sort(out_arr, axis=0)
    idx_c05 = int(0.05 * n_resamplings) - 1
    idx_c95 = int(0.95 * n_resamplings) - 1
    statistics = {
        "mean": np.mean(out_arr, axis=0),
        "median": np.median(out_arr, axis=0),
        "std": np.std(out_arr, axis=0),
        "minimum": out_arr[0],
        "maximum": out_arr[-1],
        "c05": out_arr[idx_c05],
        "c95": out_arr[idx_c95]
    }
    return statistics


def save_dict_as_arr(path, stat_dict, header=None):
    # save a dictionary with equal-length arrays as values as a np array
    # .txt file with a dict header defining rows

    row_names = list(stat_dict.keys())
    n_rows = len(row_names)
    rows = dict(zip(np.arange(n_rows), row_names))

    if not header:
        header = {"rows": rows}
    else:
        header["rows"] = rows

    arr = np.array([stat_dict[rows[idx]] for idx in rows])
    np.savetxt(path, arr, header=str(header))


def load_arr_as_dict(path):

    header, arr = load_arr(path)
    row_names = header["rows"]
    out = {row_names[idx]: arr[idx] for idx in row_names}
    return out


def get_coverage_str(bed):
    positions = bed.get_0_idx_positions()
    approx_max = np.round(bed.last_position, -6) + 1e6
    out = []
    icons = {
        0.01: "~",
        0.1: "-",
        0.25: "=",
        0.5: "*",
        0.75: "x",
        0.9: "X",
        1: "$"
    }
    key = ", ".join([f"({icons[key]}) cover < {key * 100}%" for key in icons])
    print(key)
    for i in np.arange(0, approx_max, 1e6, dtype=np.int64):
        n_positions = np.searchsorted(positions, i + 1e6) - \
                      np.searchsorted(positions, i)
        coverage = n_positions / 1e6
        for lim in icons:
            if coverage < lim:
                out.append(icons[lim])
                break
    out = "".join(out)
    return out


def plot_bootstraps(extra_stats, **bootstrap_dicts):
    # stats: list
    colors = cm.hsv(np.linspace(0, 0.9, len(bootstrap_dicts)))
    fig = plt.figure(1)
    ax = plt.subplot(111)
    for i, sample_id in enumerate(bootstrap_dicts):
        bootstrap_dict = bootstrap_dicts[sample_id]
        ax.plot(r, bootstrap_dict["mean"], marker='x', color=colors[i],
                label=sample_id, linewidth=2)
        for stat in extra_stats:
            ax.plot(r, bootstrap_dict[stat], color=colors[i])
    ax.set_ylim(0, )
    ax.set_xscale("log")
    ax.set_ylabel("H_2")
    ax.set_xlabel("r bin")
    ax.legend()
    fig.tight_layout()
    fig.show()


def plot(**kwargs):

    r_mids = two_locus.r_mids
    fig = plt.figure(1)
    ax = plt.subplot(111)
    for i, key in enumerate(kwargs):
        vals = kwargs[key]
        ax.plot(r_mids, vals, marker='x', label=key)
    ax.set_ylim(0, 1e-6)
    ax.set_xscale("log")
    ax.set_ylabel("H_2")
    ax.set_xlabel("r bin")
    ax.legend()
    fig.tight_layout()
    fig.show()
