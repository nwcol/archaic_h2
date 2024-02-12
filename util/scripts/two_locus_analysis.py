
#

import json
import numpy as np
import sys
from util import sample_sets
from util import one_locus
from util import two_locus


def get_header(statistic, window_id, window_dict, rows):
    header_dict = {
        "chrom": chrom,
        "window_id": window_id,
        "statistic": statistic,
        "bounds": window_dict["bounds"],
        "span": window_dict["span"],
        "n_sites": window_dict["n_sites"],
        "coverage": window_dict["coverage"],
        "right_discontinuous": window_dict["right_discontinuous"],
        "rows": rows
    }
    return str(header_dict)


def save_arr(arr, header, window_id, tag):
    file_name = f"{out_dir}/chr{chrom}_win{window_id}_{tag}.txt"
    file = open(file_name, "w")
    np.savetxt(file, arr, header=header)
    file.close()


if __name__ == "__main__":
    chrom = sys.argv[1]
    window_file_path = sys.argv[2]
    out_dir = sys.argv[3]
    window_file = open(window_file_path)
    full_window_dict = json.load(window_file)
    window_file.close()
    if chrom not in full_window_dict:
        raise ValueError(f"Wrong windows file; chr {chrom} not represented")
    windows_dict = full_window_dict[chrom]["windows"]
    sample_set = sample_sets.UnphasedSampleSet.read_chr(chrom)
    r_edges = two_locus.r_edges

    for window_id in windows_dict:
        window_dict = windows_dict[window_id]
        window = [int(x) for x in window_dict["bounds"]]
        limit_right = window_dict["right_discontinuous"]

        # pair counts
        pair_counts = two_locus.count_site_pairs(
            sample_set, r_edges, window=window, limit_right=limit_right
        )
        print(f"PAIR COUNTS IN WINDOW \t{window_id} CHR \t{chrom} COMPUTED")
        header = get_header(
            "pair_counts", window_id, window_dict, {0: "pair_counts"}
        )
        save_arr(pair_counts, header, window_id, "pair_counts")

        # two locus heterozygosity; one sample
        sample_ids = sample_set.sample_ids
        n_samples = len(sample_ids)
        n_bins = len(r_edges) - 1
        het_counts = np.zeros((n_samples, n_bins), dtype=np.float64)

        for i, sample_id in enumerate(sample_ids):
            het_counts[i] = two_locus.count_het_pairs(
                sample_set, sample_id, r_edges, window=window,
                limit_right=limit_right
            )
            print(f"H2_X \t{i} IN WINDOW \t{window_id} CHR \t{chrom} COMPUTED")

        rows = str(dict(zip(np.arange(n_samples), sample_ids)))
        header = get_header("H2_X", window_id, window_dict, rows)
        save_arr(het_counts, header, window_id, "H2_X_counts")

        # two locus heterozygosity; two samples
        sample_pairs = one_locus.enumerate_pairs(sample_ids)
        n_sample_pairs = len(sample_pairs)
        het_counts = np.zeros((n_sample_pairs, n_bins), dtype=np.float64)

        for i, sample_pair in enumerate(sample_pairs):
            het_counts[i] = two_locus.count_two_sample_het_pairs(
                sample_set, sample_pair[0], sample_pair[1], r_edges,
                window=window, limit_right=limit_right
            )
            print(f"H2_XY \t{i} IN WINDOW \t{window_id} CHR \t{chrom} COMPUTED")

        rows = str(dict(zip(np.arange(n_sample_pairs), sample_pairs)))
        header = get_header("H2_XY", window_id, window_dict, rows)
        save_arr(het_counts, header, window_id, "H2_XY_counts")
        print(f"TWO LOCUS ANALYSIS COMPLETE CHR {chrom} WINDOW {window_id}")
