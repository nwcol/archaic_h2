
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
    parser.add_argument("-b", "--n_resamplings", type=int, default=1_000)
    parser.add_argument("-n", '--sample_size', type=int, default=None)
    return parser.parse_args()


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


def main():

    in_files = [np.load(x) for x in args.in_fnames]
    sites = np.concatenate([x["sites"] for x in in_files], axis=0)
    H_counts = np.concatenate([x["H_counts"] for x in in_files], axis=1)

    """
    H_counts = np.concatenate(
        [np.concatenate([x["H_counts"] for x in in_files], axis=1),
         np.concatenate([x["Hxy_counts"] for x in in_files], axis=1),
         ], axis=0
    )
    H2_counts = np.concatenate(
        [np.concatenate([x["H2_counts"] for x in in_files], axis=1),
         np.concatenate([x["H2xy_counts"] for x in in_files], axis=1),
         ], axis=0
    )
    """
    site_pairs = np.concatenate([x["site_pairs"] for x in in_files], axis=0)
    H2_counts = np.concatenate([x["H2_counts"] for x in in_files], axis=1)
    r_bins = in_files[0]["r_bins"]
    n_bins = len(r_bins) - 1
    sample_names = in_files[0]["sample_names"]
    pair_names = in_files[0]["sample_pairs"]
    n_rows = len(sample_names) + len(pair_names)
    kwargs = dict(
        sample_names=sample_names,
        pair_names=pair_names,
        r_bins=r_bins
    )
    kwargs["n_bins"] = len(r_bins) - 1
    # bootstrap H
    H_dist = bootstrap(
        sites,
        H_counts,
        n_resamplings=args.n_resamplings,
        sample_size=args.sample_size
    )
    kwargs["H_dist"] = H_dist
    kwargs["H_mean"] = H_dist.mean(0)
    kwargs["H_cov"] = np.zeros((n_rows, n_rows))
    kwargs["H_cov"][:, :] = np.cov(H_dist, rowvar=False)
    # bootstrap H2
    kwargs["H2_dist"] = np.zeros((n_bins, args.n_resamplings, n_rows))
    kwargs["H2_cov"] = np.zeros((n_bins, n_rows, n_rows))
    for i in range(n_bins):
        H2_dist = bootstrap(
            site_pairs[:, i],
            H2_counts[:, :, i],
            n_resamplings=args.n_resamplings,
            sample_size=args.sample_size
        )
        kwargs["H2_dist"][i, :, :] = H2_dist
        kwargs["H2_cov"][i, :, :] = np.cov(H2_dist, rowvar=False)
    kwargs["H2_mean"] = kwargs["H2_dist"].mean(1)
    np.savez(args.out_fname, **kwargs)
    return 0


if __name__ == "__main__":
    args = get_args()
    main()
