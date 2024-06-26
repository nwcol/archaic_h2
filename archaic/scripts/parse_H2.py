
import argparse
import numpy as np
import time
from archaic import masks
from archaic import one_locus
from archaic import two_locus
from archaic import utils


def get_args():
    # get args
    parser = argparse.ArgumentParser()
    parser.add_argument("-v", "--vcf_fname", required=True)
    parser.add_argument("-m", "--mask_fname", required=True)
    parser.add_argument("-r", "--map_fname", required=True)
    parser.add_argument("-o", "--out_fname", required=True)
    parser.add_argument("-w", "--window")
    parser.add_argument("-W", "--window_fname", default=None)
    parser.add_argument("-b", "--r_bins")
    parser.add_argument("-B", "--r_bin_fname", default=None)
    parser.add_argument("-bp", "--bp_thresh", type=int, default=0)
    return parser.parse_args()


def count_sites():
    # count the number of sites in each window
    site_counts = np.zeros(n_windows, dtype=np.int64)
    for i, window in enumerate(windows):
        site_counts[i] = one_locus.count_sites(mask_positions, window)
    return site_counts


def count_site_pairs():
    # count the number of site pairs in recombination bins
    site_pair_counts = np.zeros((n_windows, n_bins), dtype=np.int64)
    for i, window in enumerate(windows):
        site_pair_counts[i] = two_locus.count_site_pairs(
            pos_map,
            r_bins,
            positions=mask_positions,
            window=window,
            upper_bound=right_bound,
            bp_thresh=args.bp_thresh,
            vectorized=True,
            verbose=1
        )
    return site_pair_counts


def get_H():
    # count one and two sample H
    counts = np.zeros((n_rows, n_windows), dtype=np.int64)
    for i in range(n_samples):
        for j, window in enumerate(windows):
            counts[i, j] = one_locus.count_H(
                genotypes[:, i],
                positions=vcf_pos,
                window=window
            )
    print(utils.get_time(), f"H parsed")
    for i, (i_x, i_y) in enumerate(pair_indices):
        i += n_samples
        for j, window in enumerate(windows):
            counts[i, j] = one_locus.count_Hxy(
                genotypes[:, i_x],
                genotypes[:, i_y],
                positions=vcf_pos,
                window=window
            )
    print(utils.get_time(), f"Hxy parsed")
    return counts


def get_H2():
    # count one and two sample H2
    counts = np.zeros((n_rows, n_windows, n_bins), dtype=np.int64)
    for i in range(n_samples):
        for j, window in enumerate(windows):
            counts[i, j] = two_locus.count_H2(
                genotypes[:, i],
                vcf_map,
                r_bins,
                positions=vcf_pos,
                window=window,
                upper_bound=right_bound,
                bp_thresh=args.bp_thresh,
                verbose=0
            )
    print(utils.get_time(), f"H2 parsed")
    for i, (i_x, i_y) in enumerate(pair_indices):
        i += n_samples
        for j, window in enumerate(windows):
            counts[i, j] = two_locus.count_H2xy(
                genotypes[:, i_x],
                genotypes[:, i_y],
                vcf_map,
                r_bins,
                positions=vcf_pos,
                upper_bound=right_bound,
                window=window,
                verbose=False
            )
    print(utils.get_time(), f"H2xy parsed")
    return counts


def parse(
    mask_fname,
    vcf_fname,
    map_fname,
    windows,
    bounds,
    r_bins,
    out_fname,
):

    t0 = time.time()
    mask_regions = masks.read_mask_regions(mask_fname)
    mask_pos = masks.read_mask_positions(mask_fname)
    vcf_pos, samples, genotypes = one_locus.read_vcf_file(
        vcf_fname, mask_regions=mask_regions
    )
    mask_map = two_locus.get_map_vals(map_fname, mask_pos)
    vcf_map = two_locus.get_map_vals(map_fname, vcf_pos)
    sample_names = np.array(samples)
    sample_pairs = one_locus.enumerate_pairs(samples)
    pair_names = np.array([f"{x},{y}" for x, y in sample_pairs])
    print(utils.get_time(), "files loaded")
    sites = one_locus.parse_site_counts(mask_pos, windows)
    H_counts = one_locus.parse_H_counts(genotypes, vcf_pos, windows)
    site_pairs = two_locus.parse_site_pairs(
        mask_map, mask_pos, windows, bounds, r_bins
    )
    H2_counts = two_locus.parse_H2_counts(
        genotypes, vcf_map, vcf_pos, windows, bounds, r_bins
    )
    arrs = dict(
        sample_names=sample_names,
        sample_pairs=pair_names,
        windows=windows,
        r_bins=r_bins,
        sites=sites,
        H_counts=H_counts,
        site_pairs=site_pairs,
        H2_counts=H2_counts
    )
    np.savez(out_fname, **arrs)
    t = np.round(time.time() - t0, 0)
    print(utils.get_time(), f"chromosome parsed in\t{t} s")


def main():
    args = get_args()
    if args.window:
        windows = np.array(eval(args.window))
    elif args.window_fname:
        windows = np.loadtxt(args.window_fname)
    else:
        raise ValueError("you must provide windows!")
    if windows.ndim != 2:
        raise ValueError(f"windows must be dim2, but are dim{windows.ndim}")
    if args.r_bins:
        r_bins = np.array(eval(args.r_bins))
    elif args.r_bin_fname:
        r_bins = np.loadtxt(args.r_bin_fname)
    else:
        raise ValueError("you must provide r bins!")
    if np.any(args.sample_names):
        sample_names = []
        for name in args.sample_names:
            if name in vcf_sample_names:
                sample_names.append(name)
            else:
                raise ValueError(f"sample {name} is not in .vcf!")
    else:
        sample_names = vcf_sample_names
    parse()


if __name__ == "__main__":
    main()
