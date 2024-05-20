
"""
Bootstrap H2 (in its respective recombination distance bins) and H, then save
the output distributions,
"""

import argparse
import numpy as np
from util.stats.parse import Abbrevs


unique_fields = [
    "sample_ids",
    "sample_pairs",
    "r_bins"
]


def bootstrap(site_pairs, het_arr, n_resamplings, sample_size=None):

    # rows correspond to samples, cols to windows
    n_rows, n_cols = het_arr.shape
    if not sample_size:
        sample_size = n_cols
    arr = np.zeros((n_resamplings, n_rows))
    for j in range(n_resamplings):
        sample_idx = np.random.choice(np.arange(n_cols), size=sample_size)
        het_sum = het_arr[:, sample_idx].sum(1)
        site_sum = site_pairs[sample_idx].sum()
        arr[j] = het_sum / site_sum
    return arr


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("in_fname")
    parser.add_argument("out_fname")
    parser.add_argument("-b", "--n_resamplings", type=int, default=1_000)
    parser.add_argument("-n", '--sample_size', type=int, default=None)
    args = parser.parse_args()

    # load up the archive
    archive = np.load(args.in_fname)
    r_bins = archive["r_bins"]
    n_bins = len(r_bins) - 1

    site_counts = archive[Abbrevs.site_counts]
    site_pair_counts = archive[Abbrevs.site_pair_counts]

    # concatenate along the "sample" axis
    het_pair_counts = np.concatenate(
        [
            archive[Abbrevs.two_locus_one_sample_het_counts],
            archive[Abbrevs.two_locus_two_sample_het_counts]
        ],
        axis=0
    )
    het_counts = np.concatenate(
        [
            archive[Abbrevs.one_locus_one_sample_het_counts],
            archive[Abbrevs.one_locus_two_sample_het_counts]
        ],
        axis=0
    )

    kwargs = {field: archive[field] for field in unique_fields}
    kwargs["n_bins"] = len(r_bins) - 1

    # bootstrap H2
    for i in range(n_bins):
        boot_dist = bootstrap(
            site_pair_counts[:, i], het_pair_counts[:, :, i],
            n_resamplings=args.n_resamplings, sample_size=args.sample_size
        )
        name = f"H2_bin{i}"
        kwargs[f"{name}_dist"] = boot_dist
        kwargs[f"{name}_mean"] = boot_dist.mean(0)
        kwargs[f"{name}_cov"] = np.cov(boot_dist, rowvar=False)

    # bootstrap H
    boot_dist = bootstrap(
        site_counts, het_counts,
        n_resamplings=args.n_resamplings, sample_size=args.sample_size
    )
    name = "H"
    kwargs[f"{name}_dist"] = boot_dist
    kwargs[f"{name}_mean"] = boot_dist.mean(0)
    kwargs[f"{name}_cov"] = np.cov(boot_dist, rowvar=False)

    np.savez(args.out_fname, **kwargs)
