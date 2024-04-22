
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
    args = parser.parse_args()
    #
    with open(args.window_file_name, 'r') as window_file:
        win_dicts = json.load(window_file)["windows"]
    sample_set = sample_sets.SampleSet.read(
        args.vcf_file_name, args.mask_file_name, None
    )
    sample_ids = sample_set.sample_ids
    sample_pairs = sample_set.sample_pairs
    chrom = sample_set.chrom
    row_names = [f"chr{chrom}_win{x}" for x in win_dicts]
    window_tuples = [(chrom, win_dicts[x]["bounds"]) for x in win_dicts]
    het_counts = {sample_id: [] for sample_id in sample_ids}
    het_xy_counts = {sample_pair: [] for sample_pair in sample_pairs}
    #
    for i, window_tuple in enumerate(window_tuples):
        window = window_tuple[1]
        for sample_id in sample_ids:
            het_counts[sample_id].append(
                sample_set.count_het_sites(sample_id, window=window)
            )
        for sample_pair in sample_pairs:
            het_xy_counts[sample_pair].append(
                sample_set.het_xy(
                    sample_pair[0], sample_pair[1], window=window
                )
            )
        print(f"het. site counts parsed\twin{i}\tchr{chrom}")
    #
    arr = np.array(
        [het_counts[sample_id] for sample_id in sample_ids] +
        [het_xy_counts[sample_pair] for sample_pair in sample_pairs]
    ).T
    header = dict(
        chrom=chrom,
        statistic="het_counts",
        sample_ids=sample_ids + sample_pairs,
        windows=window_tuples,
        vcf_file=args.vcf_file_name,
        window_file=args.window_file_name,
        cols=sample_ids + sample_pairs,
        rows=row_names
    )
    file_util.save_arr(args.out_file_name, arr, header=header)
    #
    print(f"het. counts parsed\tchr{chrom}")
