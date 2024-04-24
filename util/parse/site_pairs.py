
"""
Parse binned site pair counts across chromosomes.

For my own convenience, arguments are paths to directories, where this file
structure is assumed to exist:
x_dir/
    chr1*.x
    chr2*.x
    ...
    chr22*.x
"""

import argparse
import json
import numpy as np
from util import file_util
from util import masks
from util import maps
from util import two_locus


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("bed_file_name")
    parser.add_argument("map_file_name")
    parser.add_argument("window_file_name")
    parser.add_argument("out_file_name")
    parser.add_argument("-r", "--r_bin_file")
    parser.add_argument("-t", "--bp_threshold", type=int, default=0)
    args = parser.parse_args()
    #
    if args.r_bin_file:
        r_edges = np.loadtxt(args.r_bin_file)
    else:
        r_edges = two_locus.r_edges
    with open(args.window_file_name, 'r') as window_file:
        win_dicts = json.load(window_file)["windows"]
    mask = masks.Bed.read_bed(args.bed_file_name)
    positions = mask.positions_1
    genetic_map = maps.GeneticMap.read_txt(args.map_file_name)
    chrom = mask.chrom
    rows = []
    windows = []
    pair_counts = []
    #
    for win_id in win_dicts:
        win_dict = win_dicts[win_id]
        bounds = win_dict["bounds"]
        lim_right = win_dict["limit_right"]
        rows.append(f"chr{chrom}_win{win_id}")
        windows.append((chrom, bounds))
        pair_counts.append(
            two_locus.count_site_pairs(
                positions,
                genetic_map,
                r_edges,
                window=bounds,
                limit_right=lim_right,
                bp_threshold=args.bp_threshold
            )
        )
        print(f"site pair counts parsed\twin{win_id}\tchr{chrom}")
    #
    n_r_bins = len(r_edges) - 1
    cols = [f"r_({r_edges[b]}, {r_edges[b + 1]})" for b in range(n_r_bins)]
    header = dict(
        chrom=chrom,
        statistic="site_pair_counts",
        windows=windows,
        vcf_file=args.vcf_file_name,
        bed_file=args.bed_file_name,
        map_file=args.map_file_name,
        window_file=args.window_file_name,
        cols=cols,
        rows=rows
    )
    file_util.save_arr(args.out_file_name, pair_counts, header)
    print(f"site pair counts parsed on \tchr{chrom}")
