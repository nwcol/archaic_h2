
"""
Parse binned site pair counts from a series of windows across one chromosome
"""

import argparse
import json
import numpy as np
from util import file_util
from util import sample_sets
from util import two_locus


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("vcf_file_name")
    parser.add_argument("bed_file_name")
    parser.add_argument("map_file_name")
    parser.add_argument("window_file")
    parser.add_argument("out_file_name")
    parser.add_argument("-t", "--bp_threshold", type=int, default=0)
    parser.add_argument("-r", "--r_bin_file", nargs='*', type=float)
    args = parser.parse_args()
    #
    if args.r_bin_file:
        r_edges = np.loadtxt(args.r_bin_file)
    else:
        r_edges = two_locus.r_edges
    with open(args.window_file, 'r') as window_file:
        window_dicts = json.load(window_file)["windows"]
    sample_set = sample_sets.USampleSet.read(
        args.vcf_file_name, args.bed_file_name, args.map_file_name
    )
    chrom = sample_set.chrom
    row_names = []
    pair_counts = {}
    #
    for window_id in window_dicts:
        window_dict = window_dicts[window_id]
        bounds = window_dict["bounds"]
        limit_right = window_dict["limit_right"]
        row_name = f"chr{chrom}_win{window_id}"
        row_names.append(row_name)
        pair_counts[row_name] = two_locus.count_site_pairs(
            sample_set,
            r_edges,
            window=bounds,
            limit_right=limit_right,
            bp_threshold=args.bp_threshold
        )
        count_dict = {"pair_counts": pair_counts}
        print(f"SITE PAIRS COUNTED IN\tWIN{window_id}\tCHR{chrom}")
    #
    n_r_bins = len(r_edges) - 1
    cols = {b: f"r_({r_edges[b]}, {r_edges[b + 1]})" for b in range(n_r_bins)}
    header = file_util.get_header(
        chrom=chrom,
        statistic="site_pair_counts",
        windows=window_dicts,
        vcf_file=args.vcf_file_name,
        bed_file=args.bed_file_name,
        map_file=args.map_file_name,
        bp_threshold=args.bp_threshold,
        cols=cols
    )
    file_util.save_dict_as_arr(args.out_file_name, pair_counts, header)
    print(f"SITE PAIRS COUNTED ON\tCHR{chrom}")
