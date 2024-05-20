
"""
Perform a bootstrap on statistics distributed across several files.
"""


import argparse
import numpy as np
from util import file_util


def bootstrap(norm_vec, stat_arr, n_resamplings, sample_size=None):
    """
    Perform a boostrap by resampling rows from stat_arr randomly with
    replacement.

    :param norm_vec:
    :param stat_arr:
    :param n_resamplings:
    :param sample_size:
    :return:
    """
    n_rows, n_cols = stat_arr.shape
    if not sample_size:
        sample_size = n_rows
    boot_arr = np.zeros((n_resamplings, n_cols))
    for i in np.arange(n_resamplings):
        sample_idx = np.random.choice(np.arange(n_rows), size=sample_size)
        stat_sums = stat_arr[sample_idx].sum(0)
        norm_sum = norm_vec[sample_idx].sum()
        boot_arr[i] = stat_sums / norm_sum
    return boot_arr


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("norm_file_name")
    parser.add_argument("bin_idx", type=int)
    parser.add_argument("stat_file_names", nargs='*')
    parser.add_argument("-o", "--out_file_prefix", default=None)
    parser.add_argument("-b", "--n_resamplings", type=int, default=1_000)
    parser.add_argument("-n", '--sample_size', type=int, default=None)
    args = parser.parse_args()

    header, norm_arr = file_util.load_arr(args.norm_file_name)
    norm_vec = norm_arr[:, args.bin_idx]

    # load indexed column from each specified file
    stats = {}
    for i, file_name in enumerate(args.stat_file_names):
        header, arr = file_util.load_arr(file_name)
        sample_id = header["sample_id"]
        if "," in sample_id:
            split = sample_id.split(",")
            sample_id = (split[0], split[1])
        stats[sample_id] = arr[:, args.bin_idx]

    # sort stats into an array
    stat_names = list(stats.keys())
    stat_arr = np.array([stats[x] for x in stat_names], dtype=np.float64).T

    # preform the bootstrap
    boot_arr = bootstrap(norm_vec, stat_arr, args.n_resamplings)

    out_header = file_util.get_header(
        bin_idx=args.bin_idx,
        bin_name=header["cols"][args.bin_idx],
        n_resamplings=args.n_resamplings,
        cols=stat_names
    )
    if not args.out_file_prefix:
        prefix = f"bin{args.bin_idx}"
    else:
        prefix = args.out_file_prefix
    dist_file_name = f"{prefix}_dist.txt"
    mean_file_name = f"{prefix}_mean.txt"
    cov_file_name = f"{prefix}_cov.txt"

    file_util.save_arr(dist_file_name, boot_arr, out_header)
    file_util.save_arr(mean_file_name, boot_arr.mean(0), out_header)

    # compute covariance matrix and save
    cov_matrix = np.cov(boot_arr, rowvar=False)
    file_util.save_arr(cov_file_name, cov_matrix, out_header)
