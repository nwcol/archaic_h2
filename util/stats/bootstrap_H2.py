
"""
Bootstrap H2 (in its respective recombination distance bins) and H, then save
the output distributions,
"""

import argparse
import numpy as np


unique_fields = [
    "sample_names",
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

    site_counts = archive["site_counts"]
    site_pair_counts = archive["site_pair_counts"]
    # concatenate along the "sample" axis
    het_pair_counts = np.concatenate(
        [archive["H2_counts"], archive["H2xy_counts"]], axis=0
    )
    het_counts = np.concatenate(
        [archive["H_counts"], archive["Hxy_counts"]], axis=0
    )
    kwargs = {field: archive[field] for field in unique_fields}
    kwargs["n_bins"] = len(r_bins) - 1
    # bootstrap H2
    for i in range(n_bins):
        boot_dist = bootstrap(
            site_pair_counts[:, i], het_pair_counts[:, :, i],
            n_resamplings=args.n_resamplings, sample_size=args.sample_size
        )
        kwargs[f"H2_bin{i}_dist"] = boot_dist
        kwargs[f"H2_bin{i}_mean"] = boot_dist.mean(0)
        kwargs[f"H2_bin{i}_cov"] = np.cov(boot_dist, rowvar=False)
    # bootstrap H
    boot_dist = bootstrap(
        site_counts, het_counts,
        n_resamplings=args.n_resamplings, sample_size=args.sample_size
    )
    kwargs[f"H_dist"] = boot_dist
    kwargs[f"H_mean"] = boot_dist.mean(0)
    kwargs[f"H_cov"] = np.cov(boot_dist, rowvar=False)
    np.savez(args.out_fname, **kwargs)
