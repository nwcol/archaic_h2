
import argparse
import numpy as np
from util import file_util


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("boot_file_names", nargs='*')
    parser.add_argument("-v", "--cov_matrix_file_names", nargs='*')
    args = parser.parse_args()
    #
    boot_arrs = {}
    for file_name in args.boot_file_names:
        boot_header, boot_arr = file_util.load_arr(file_name)
        col_idx = boot_header["col_idx"]
        boot_arrs[col_idx] = boot_arr
    # means
    col_idxs = list(boot_arrs.keys())
    col_idxs.sort()
    boot_means = [boot_arrs[i].mean(0) for i in col_idxs]
    means_arr = np.array(boot_means).T
    boot_header = file_util.read_header(args.boot_file_names[0])
    header = {"rows": boot_header["cols"]}
    file_util.save_arr("means.txt", means_arr, header)
    # stds
    cov_matrices = {}
    for file_name in args.cov_matrix_file_names:
        cov_header, cov_matrix = file_util.load_arr(file_name)
        col_idx = cov_header["col_idx"]
        cov_matrices[col_idx] = cov_matrix
    n = len(cov_matrices[0])
    b = len(cov_matrices)
    var_arr = np.zeros((n, b))
    col_idxs = list(cov_matrices.keys())
    for i in col_idxs:
        cov_matrix = cov_matrices[i]
        var_arr[:, i] = cov_matrix[np.arange(n), np.arange(n)]
    std_arr = np.sqrt(var_arr)
    file_util.save_arr("stds.txt", std_arr, header)
