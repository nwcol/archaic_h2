"""
Remove some bins from a bootstrap archive

start is an inclusive bin index; stop is noninclusive
eg 10:18 selects bins 10 to 17 corresponding to r edges 10 to 18
"""

import argparse
import numpy as np


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("in_archive_name")
    parser.add_argument("out_archive_name")
    parser.add_argument("start", type=int)
    parser.add_argument("stop", type=int)
    args = parser.parse_args()

    in_archive = np.load(args.in_archive_name)
    r_bins = in_archive["r_bins"]
    sliced_r = r_bins[args.start:args.stop + 1]
    n_bins = len(sliced_r) - 1
    print(f"returning {n_bins} r bins {sliced_r.min()} to "
          f"{sliced_r.max()}:\n{sliced_r}")
    kwargs = {
        "n_bins": n_bins,
        "r_bins": sliced_r,
        "sample_ids": in_archive["sample_ids"],
        "sample_pairs": in_archive["sample_pairs"],
        "H_dist": in_archive["H_dist"],
        "H_mean": in_archive["H_mean"],
        "H_cov": in_archive["H_cov"]
    }
    for k, i in enumerate(np.arange(args.start, args.stop)):
        for field in ["dist", "mean", "cov"]:
            kwargs[f"H2_bin{k}_{field}"] = in_archive[f"H2_bin{i}_{field}"]

    np.savez(args.out_archive_name, **kwargs)
