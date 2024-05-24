
"""
Pack chromosome .npz output files into a single .npz file
"""

import argparse
import numpy as np


unique_fields = [
    "sample_names",
    "sample_pairs",
    "r_bins"
]

window_idx0_fields = [
    "chroms",
    "windows",
    "site_counts",
    "site_pair_counts"
]

window_idx1_fields = [
    "H_counts",
    "Hxy_counts",
    "H2_counts",
    "H2xy_counts"
]


def get_args():

    parser = argparse.ArgumentParser()
    parser.add_argument("out_fname")
    parser.add_argument("in_fnames", nargs='*')
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    # sort files by increasing chromosome number
    numbers = [np.load(fname)["chroms"][0] for fname in args.in_fnames]
    sorted_numbers = sorted(numbers)
    searchsort_idx = [numbers.index(n) for n in sorted_numbers]
    in_fnames = [args.in_fnames[i] for i in searchsort_idx]
    archives = [np.load(fname) for fname in in_fnames]
    archive_0 = archives[0]
    fields = archive_0.files
    # create kwargs dictionary
    kwargs = {field: archive_0[field] for field in unique_fields}
    for field in window_idx0_fields:
        if field in fields:
            kwargs[field] = np.concatenate(
                [archive[field] for archive in archives], axis=0
            )
    for field in window_idx1_fields:
        if field in fields:
            kwargs[field] = np.concatenate(
                [archive[field] for archive in archives], axis=1
            )
    # computing statistics
    windows = kwargs["windows"]
    n_windows = len(windows)
    chroms = kwargs["chroms"]
    n_chroms = len(chroms)
    # compute one-locus statistics
    if "site_counts" in fields:
        site_counts = kwargs["site_counts"]
        kwargs["n_sites"] = site_counts.sum()
        for stat in ["H", "Hxy"]:
            if f"{stat}_counts" in fields:
                het = kwargs[f"{stat}_counts"]
                kwargs[f"window_{stat}"] = het / site_counts
                kwargs[f"chrom_{stat}"] = np.array([
                    het[:, chroms == c].sum(1) / site_counts[chroms == c].sum()
                    for c in np.unique(chroms)
                ]).T
                kwargs[stat] = het.sum(1) / site_counts.sum()
    # compute two-locus statistics
    if "site_pair_counts" in fields:
        site_pairs = kwargs["site_pair_counts"]
        for stat in ["H2", "H2xy"]:
            if f"{stat}_counts" in fields:
                het = kwargs[f"{stat}_counts"]
                kwargs[f"window_{stat}"] = het / site_pairs
                kwargs[f"chrom_{stat}"] = np.array([
                    het[:, chroms == c].sum(1) / site_pairs[chroms == c].sum(0)
                    for c in np.unique(chroms)
                ]).T
                kwargs[stat] = het.sum(1) / site_pairs.sum(0)
    np.savez(args.out_fname, **kwargs)
