"""

"""


import argparse
import numpy as np


unique_fields = [
    "sample_names",
    "sample_pairs",
    "r_bins"
]


def get_args():
    # get args
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--in_fnames", nargs='*', required=True)
    parser.add_argument("-o", "--out_fname", required=True)
    parser.add_argument("-n", "--n_iters", type=int, default=1_000)
    return parser.parse_args()


def bootstrap_H2(pair_counts, H2_counts, n_iters=1000):
    #
    n, n_windows, n_bins = H2_counts.shape
    arr = np.zeros((n_iters, n_bins, n))
    for i in range(n_iters):
        idx = np.random.randint(n_windows, size=n_windows)
        H2_sum = H2_counts[:, idx, :].sum(1)
        pair_sum = pair_counts[idx, :].sum(0)
        arr[i] = (H2_sum / pair_sum).T
    return arr


def bootstrap_H(site_counts, H_counts, n_iters=1000):
    #
    n, n_windows = H_counts.shape
    arr = np.zeros((n_iters, n))
    for i in range(n_iters):
        idx = np.random.randint(n_windows, size=n_windows)
        H_sum = H_counts[:, idx].sum(1)
        site_sum = site_counts[idx].sum()
        arr[i] = H_sum / site_sum
    return arr


def bootstrap(in_fnames, out_fname, n_iters=1000):
    #
    in_files = [np.load(x) for x in in_fnames]
    sites = np.concatenate([x["site_counts"] for x in in_files], axis=0)
    H_counts = np.concatenate([x["H_counts"] for x in in_files], axis=1)
    site_pairs = np.concatenate([x["pair_counts"] for x in in_files], axis=0)
    H2_counts = np.concatenate([x["H2_counts"] for x in in_files], axis=1)
    r_bins = in_files[0]["r_bins"]
    n_bins = len(r_bins) - 1
    sample_names = in_files[0]["sample_names"]
    pair_names = in_files[0]["sample_pairs"]
    n = len(sample_names) + len(pair_names)
    H_dist = bootstrap_H(sites, H_counts, n_iters=n_iters)
    H_mean = H_dist.mean(0)
    H_cov = np.cov(H_dist, rowvar=False)
    H2_dist = bootstrap_H2(site_pairs, H2_counts, n_iters)
    H2_cov = np.zeros((n_bins, n, n))
    for i in range(n_bins):
        H2_cov[i, :, :] = np.cov(H2_dist[:, i, :], rowvar=False)
    H2_mean = H2_dist.mean(0)
    arrs = dict(
        sample_names=sample_names,
        pair_names=pair_names,
        r_bins=r_bins,
        H_dist=H_dist,
        H_mean=H_mean,
        H_cov=H_cov,
        H2_dist=H2_dist,
        H2_mean=H2_mean,
        H2_cov=H2_cov
    )
    np.savez(out_fname, **arrs)


def main():
    #
    args = get_args()
    bootstrap(args.in_fnames, args.out_fname, n_iters=args.n_iters)
    return 0


if __name__ == "__main__":
    main()
