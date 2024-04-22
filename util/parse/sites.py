
"""
Count the number of sites represented in a series of windows on one chromosome
"""

import argparse
import json
import numpy as np
from util import bed_util
from util import file_util


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("bed_file_name")
    parser.add_argument("window_file_name")
    parser.add_argument("out_file_name")
    args = parser.parse_args()
    #
    with open(args.window_file_name, 'r') as window_file:
        win_dicts = json.load(window_file)["windows"]
    bed = bed_util.Bed.read_bed(args.bed_file_name)
    chrom = bed.chrom
    positions = bed.positions_1
    rows = []
    windows = []
    site_counts = []
    #
    for win_id in win_dicts:
        win_dict = win_dicts[win_id]
        bounds = win_dict["bounds"]
        lim_right = win_dict["limit_right"]
        rows.append(f"chr{chrom}_win{win_id}")
        windows.append((chrom, bounds))
        site_counts.append(
            np.diff(np.searchsorted(positions, bounds))
        )
        print(f"site counts parsed\twin{win_id}\tchr{chrom}")
    #
    cols = ["site_counts"]
    header = dict(
        chrom=chrom,
        statistic="site_counts",
        windows=windows,
        bed_file=args.bed_file_name,
        window_file=args.window_file_name,
        cols=cols,
        rows=rows
    )
    file_util.save_arr(args.out_file_name, site_counts, header)
    print(f"site counts parsed\tchr{chrom}")
