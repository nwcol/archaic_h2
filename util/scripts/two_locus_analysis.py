
#

import json
import numpy as np
import sys
from util import sample_sets
from util import two_locus


if __name__ == "__main__":
    chrom = sys.argv[1]
    window_file_path = sys.argv[2]
    out_dir = sys.argv[3]
    window_file = open(window_file_path)
    window_dict = json.load(window_file)
    window_file.close()
    windows = window_dict[chrom]
    sample_set = sample_sets.UnphasedSampleSet.get_chr(chrom)
    r_edges = np.array([0, 1e-7, 2e-7, 5e-7, 1e-6, 2e-6, 5e-6, 1e-5, 2e-5,
                        5e-5, 1e-4, 2e-4, 5e-4, 1e-3, 2e-3, 5e-3, 1e-2, 2e-2,
                        5e-2, 1e-1], dtype=np.float64)


    def main(window, window_id):
        window = [int(x) for x in window]
        n_positions = sample_set.interval_n_positions(window[0], window[1])

        # one-sample
        pair_counts = two_locus.count_site_pairs(
            sample_set, r_edges, window=window
        )
        sample_ids = sample_set.sample_ids
        n_samples = len(sample_ids)
        n_bins = len(r_edges) - 1
        two_locus_het = np.zeros((n_samples, n_bins), dtype=np.float64)
        for i, sample_id in enumerate(sample_ids):
            het_counts = two_locus.count_het_pairs(
                sample_set, sample_id, r_edges, window=window
            )
            two_locus_het[i] = het_counts / pair_counts
            print(f"H2 {i} window {window_id} computed")
        header_dict = {
            "chr": sample_set.chrom,
            "stat": "H2",
            "window": window,
            "window_id": window_id,
            "n_sites": n_positions,
            "rows": str(dict(zip(np.arange(n_samples), sample_ids)))
        }
        header = str(header_dict)
        file = open(f"{out_dir}/chr{chrom}_win{window_id}_H2.txt", "w")
        np.savetxt(file, two_locus_het, header=header)
        file.close()

        # two-sample
        sample_pairs = two_locus.enumerate_pairs(sample_ids)
        n_sample_pairs = len(sample_pairs)
        two_locus_cross_het = np.zeros((n_sample_pairs, n_bins), dtype=np.float64)
        for i, sample_pair in enumerate(sample_pairs):
            het_counts = two_locus.count_cross_pop_het_pairs(
                sample_set, sample_pair[0], sample_pair[1], r_edges, window=window
            )
            two_locus_cross_het[i] = het_counts / pair_counts
            print(f"H2_2 {i} window {window_id} computed")
        header_dict = {
            "chr": sample_set.chrom,
            "stat": "H2_2",
            "window": window,
            "window_id": window_id,
            "n_sites": n_positions,
            "rows": str(dict(zip(np.arange(n_sample_pairs), sample_pairs)))
        }
        cross_header = str(header_dict)
        file = open(f"{out_dir}/chr{chrom}_win{window_id}_H2_2.txt", "w")
        np.savetxt(file, two_locus_cross_het, header=cross_header)
        file.close()
        print(f"two locus analysis complete; chr{chrom} win{window_id}")


    #for _window_id in windows:
    #    _window = windows[_window_id]["limits"]
    #    main(_window, _window_id)
    main(windows['1']["limits"], "1")
