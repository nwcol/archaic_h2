
import argparse
import json
import numpy as np
import sys
from util import file_util
from util import sample_sets
from util import two_locus


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("chrom")
    parser.add_argument('-b', "--window_bounds")
    parser.add_argument("-w", "--window_file")
    parser.add_argument("-o", "--output_file")
    parser.add_argument("-t", "--bp_threshold", type=int)
    #
    args = parser.parse_args()
    #
    sample_set = sample_sets.UnphasedSampleSet.read_chr(args.chrom)
    #
    if args.window_bounds:
        window_bounds = eval(args.window_bounds)
        window_dict = {i:
            {"bounds": (int(win0), int(win1))}
                for i, (win0, win1) in enumerate(window_bounds)
        }
    elif args.window_file:
        window_dict = json.load(args.window_file)[args.chrom]["windows"]
    else:
        print("YOU MUST PROVIDE A WINDOW FILE OR LIST OF WINDOWS")
        sys.exit()
    #
    if args.bp_threshold:
        bp_threshold = args.bp_threshold
    else:
        bp_threshold = 0
    #
    for window_id in window_dict:
        window = window_dict[window_id]
        window_bounds = [int(x) for x in window["bounds"]]
        if "right_discontinuous" in window:
            limit_right = window["right_discontinuous"]
        else:
            limit_right = False
        #
        pair_counts = two_locus.count_site_pairs(
            sample_set, two_locus.r_edges, window=window_bounds,
            limit_right=limit_right, bp_threshold=bp_threshold
        )
        #
        print(f"SITE PAIRS COUNTED IN \t WINDOW {window_id} "
              f"\tCHR{args.chrom}")
        header = file_util.get_header(
            "pair_counts", window_id, window_dict, {0: "pair_counts"}
        )
        pair_counts = pair_counts[np.newaxis, :]
        if args.output_file:
            file_util.save_arr(args.output_file, pair_counts, header)
        else:
            print(header, "\n", pair_counts)

    print(f"SITE PAIRS COUNTED ON \tCHR{args.chrom}")
