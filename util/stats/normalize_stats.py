
# take an array of statistics (eg. heterozygous site counts) and normalize
# them by an array of the same size (eg. site counts)

import argparse
import numpy as np
from util import file_util


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("in_file_name")
    parser.add_argument("norm_file_name")
    parser.add_argument("out_file_name")
    parser.add_argument("-s", "--statistic")
    args = parser.parse_args()
    #
    norm_header, norm_arr = file_util.load_arr(args.norm_file_name)
    norm_sum = np.sum(norm_arr, axis=0)
    file_header, arr = file_util.load_arr(args.in_file_name)
    normalized_arr = arr / norm_arr
    if args.statistic:
        file_header["statistic"] = args.statistic
    else:
        file_header["statistic"] = None
    file_util.save_arr(args.out_file_name, normalized_arr, file_header)
