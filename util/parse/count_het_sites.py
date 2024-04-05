
"""
Count the number of heterozygous sites in a series of windows on one chromosome
"""

import argparse
import json
import numpy as np
from util import file_util
from util import sample_sets


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("vcf_file_name")
    parser.add_argument("mask_file_name")
    parser.add_argument("window_file_name")
    parser.add_argument("out_file_name")
    parser.add_argument("-s", "--sample_ids", nargs='*', default=None)
    args = parser.parse_args()
    #
    with open(args.window_file_name, 'r') as window_file:
        win_dicts = json.load(window_file)["windows"]
    sample_set = sample_sets.USampleSet.read(
        args.vcf_file_name, args.mask_file_name, None
    )
    chrom = sample_set.chrom
    if args.sample_ids:
        sample_ids = args.sample_ids
    else:
        sample_ids = sample_set.sample_ids
    rows = []
    windows = []
    het_counts = {sample_id: [] for sample_id in sample_ids}
    #
    for win_id in win_dicts:
        win_dict = win_dicts[win_id]
        bounds = win_dict["bounds"]
        lim_right = win_dict["limit_right"]
        rows.append(f"chr{chrom}_win{win_id}")
        windows.append((chrom, bounds))
        for sample_id in sample_ids:
            het_counts[sample_id].append(
                sample_set.count_het_sites(sample_id, bounds=bounds)
            )
        print(f"het. site counts parsed\twin{win_id}\tchr{chrom}")
    #
    arr = np.array([het_counts[sample_id] for sample_id in sample_ids]).T
    header = dict(
        chrom=chrom,
        statistic="site_counts",
        sample_ids=sample_ids,
        windows=windows,
        vcf_file=args.vcf_file_name,
        window_file=args.window_file_name,
        cols=sample_ids,
        rows=rows
    )
    file_util.save_arr(args.out_file_name, arr, header=header)
    #
    print(f"het. counts parsed\tchr{chrom}")
