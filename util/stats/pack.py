
"""
Pack chromosome .npz output files into a single .npz file

Emits (X, Y are sample names)
    r_bins
    chroms
    windows

    sites               n_sites
    site_pairs          n_site_pairs
    hets_X
    hets_X,Y
    het_pairs_X
    het_pairs_X,Y

    H_X_arr             H_X
    H_X,Y_arr           H_X,Y
    H2_X_arr            H2_X
    H2_X,Y_arr          H2_X,Y
"""

import argparse
import numpy as np
from util.stats.parse import Abbrevs


unique_fields = [
    "sample_ids",
    "sample_pairs",
    "r_bins"
]

window_idx0_fields = [
    "chroms",
    "windows",
    Abbrevs.site_counts,
    Abbrevs.site_pair_counts
]

window_idx1_fields = [
    Abbrevs.one_locus_one_sample_het_counts,
    Abbrevs.one_locus_two_sample_het_counts,
    Abbrevs.two_locus_one_sample_het_counts,
    Abbrevs.two_locus_two_sample_het_counts
]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("out_fname")
    parser.add_argument("in_fnames", nargs='*')
    args = parser.parse_args()

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

    # compute statistics in windows, chromosomes, and genome-wide



    np.savez(args.out_fname, **kwargs)
