
import argparse
import numpy as np
from util import file_util


def bootstrap(norm_vec, stat_arr, n_resamplings, sample_size=None):

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


statistic_name_map = {
    "het_pair_counts": "H_2_X",
    "two_sample_het_pair_counts": "H_2_XY"
}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("norm_file_name")
    parser.add_argument("col_idx", type=int)
    parser.add_argument("dist_file_name")
    parser.add_argument("mean_file_name")
    parser.add_argument("cov_file_name")
    parser.add_argument("stat_file_names", nargs='*')
    parser.add_argument("-b", "--n_resamplings", type=int, default=1_000)
    parser.add_argument("-n", '--sample_size', type=int, default=None)
    args = parser.parse_args()
    #
    header, norm_arr = file_util.load_arr(args.norm_file_name)
    norm_vec = norm_arr[:, args.col_idx]
    stats = {}
    #
    for i, file_name in enumerate(args.stat_file_names):
        header, arr = file_util.load_arr(file_name)
        sample_id = header["sample_id"]
        if "," in sample_id:
            split = sample_id.split(",")
            sample_id = (split[0], split[1])
        stats[sample_id] = arr[:, args.col_idx]
    #
    stat_names = list(stats.keys())
    stat_arr = np.array([stats[x] for x in stat_names], dtype=np.float64).T
    boot_arr = bootstrap(norm_vec, stat_arr, args.n_resamplings)
    boot_header = file_util.get_header(
        col_idx=args.col_idx,
        col_name=header["cols"][args.col_idx],
        n_resamplings=args.n_resamplings,
        cols=stat_names
    )
    file_util.save_arr(args.dist_file_name, boot_arr, boot_header)
    file_util.save_arr(args.mean_file_name, boot_arr.mean(0), boot_header)
    #
    cov_matrix = np.cov(boot_arr, rowvar=False)
    cov_header = file_util.get_header(
        col_idx=args.col_idx,
        col_name=header["cols"][args.col_idx],
        n_resamplings=args.n_resamplings,
        cols=stat_names,
        rows=stat_names
    )
    file_util.save_arr(args.cov_file_name, cov_matrix, cov_header)
