
"""
Parse statistics from a chromosome. Chromosome representations are constructed
from three input types; a .vcf defining variant sites, a .bed mask defining
coverage, and a .txt map file defining the recombination landscape.
"""

import argparse
import numpy as np
import time
from util import one_locus
from util import two_locus


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-v", "--vcf_fname", required=True)
    parser.add_argument("-m", "--mask_fname", required=True)
    parser.add_argument("-r", "--map_fname", required=True)
    parser.add_argument("-o", "--out_fname", required=True)

    parser.add_argument("-n", "--chromosome_number", type=int)
    parser.add_argument("-w", "--window")
    parser.add_argument("-W", "--window_fname", default=None)
    parser.add_argument("-b", "--r_bins")
    parser.add_argument("-B", "--r_bin_fname", default=None)
    parser.add_argument("-s", "--sample_names", default=None, nargs='*')
    parser.add_argument("-bp", "--bp_thresh", type=int, default=0)

    parser.add_argument("-c", "--parse_site_counts", type=int, default=1)
    parser.add_argument("-1l", "--parse_one_locus", type=int, default=1)
    parser.add_argument("-2l", "--parse_two_locus", type=int, default=1)
    parser.add_argument("-1s", "--parse_one_sample", type=int, default=1)
    parser.add_argument("-2s", "--parse_two_sample", type=int, default=1)
    return parser.parse_args()


def window_discontinuity(windows):
    """
    Return a vector indicating whether the "lim_right" pair-counting parameter
    should be true or false (it should be false unless the window is the last
    window, or there is a discontinuity between it and the next window start)

    :param windows:
    :return:
    """
    n_windows = len(windows)
    indicator = np.full(n_windows, 0)
    for i in range(n_windows - 1):
        if windows[i, 1] < windows[i+1, 0]:
            indicator[i] = 1
    indicator[-1] = 1
    return indicator


def count_sites():

    site_counts = np.zeros(n_windows, dtype=np.int64)
    for i, window in enumerate(windows):
        site_counts[i] = one_locus.count_sites(mask_positions, window)
    kwargs["site_counts"] = site_counts
    kwargs["n_sites"] = site_counts.sum()


def count_site_pairs():

    site_pair_counts = np.zeros((n_windows, n_bins), dtype=np.int64)
    for i, window in enumerate(windows):
        site_pair_counts[i] = two_locus.count_site_pairs(
            pos_map,
            r_bins,
            positions=mask_positions,
            window=window,
            bp_thresh=args.bp_thresh
        )
    kwargs["site_pair_counts"] = site_pair_counts
    kwargs["n_site_pairs"] = site_pair_counts.sum()


def get_H():

    counts = np.zeros((n_samples, n_windows), dtype=np.int64)
    for i in range(n_samples):
        for j, window in enumerate(windows):
            counts[i, j] = one_locus.count_H(
                genotypes[:, i],
                positions=vcf_pos,
                window=window
            )
    kwargs["H_counts"] = counts


def get_Hxy():

    counts = np.zeros((n_sample_pairs, n_windows), dtype=np.float64)
    for i, (i_x, i_y) in enumerate(pair_indices):
        for j, window in enumerate(windows):
            counts[i, j] = one_locus.count_Hxy(
                genotypes[:, i_x],
                genotypes[:, i_y],
                positions=vcf_pos,
                window=window
            )
    kwargs["Hxy_counts"] = counts


def get_H2():

    counts = np.zeros((n_samples, n_windows, n_bins), dtype=np.int64)
    for i in range(n_samples):
        for j, window in enumerate(windows):
            counts[i, j] = two_locus.count_H2(
                genotypes[:, i],
                vcf_map,
                r_bins,
                positions=vcf_pos,
                window=window,
                bp_thresh=args.bp_thresh
            )
    kwargs["H2_counts"] = counts


def get_H2xy():

    counts = np.zeros((n_sample_pairs, n_windows, n_bins), dtype=np.float64)
    for i, (i_x, i_y) in enumerate(pair_indices):
        for j, window in enumerate(windows):
            counts[i, j] = two_locus.count_H2xy(
                genotypes[:, i_x],
                genotypes[:, i_y],
                vcf_map,
                r_bins,
                positions=vcf_pos,
                window=window
            )
    kwargs["H2xy_counts"] = counts


if __name__ == "__main__":
    t0 = time.time()
    args = get_args()
    if args.window:
        windows = np.array(eval(args.window))
    elif args.window_fname:
        windows = np.loadtxt(args.window_fname)
    else:
        raise ValueError("you must provide windows!")
    if windows.ndim != 2:
        raise ValueError(f"windows must be dim2, but are dim{windows.ndim}")
    right_lims = window_discontinuity(windows)

    if args.parse_two_locus:
        if args.r_bins:
            r_bins = np.array(eval(args.r_bins))
        elif args.r_bin_fname:
            r_bins = np.loadtxt(args.r_bin_fname)
        else:
            raise ValueError("you must provide r bins!")
        n_bins = len(r_bins) - 1
    else:
        r_bins = None
        n_bins = 0

    mask_regions = one_locus.read_mask_regions(args.mask_fname)
    mask_positions = one_locus.mask_positions_from_regions(mask_regions)
    vcf_pos, vcf_sample_names, genotypes = one_locus.read_vcf_file(
        args.vcf_fname, mask_regions=mask_regions
    )
    pos_map = two_locus.get_map_vals(args.map_fname, mask_positions)
    vcf_map = two_locus.get_map_vals(args.map_fname, vcf_pos)

    if np.any(args.sample_names):
        sample_names = []
        for name in args.sample_names:
            if name in vcf_sample_names:
                sample_names.append(name)
            else:
                raise ValueError(f"sample {name} is not in .vcf!")
    else:
        sample_names = vcf_sample_names

    sample_arr = np.array(sample_names)
    n_samples = len(sample_names)

    sample_pairs = one_locus.enumerate_pairs(sample_names)
    n_sample_pairs = len(sample_pairs)
    pair_indices = one_locus.enumerate_indices(n_samples)
    sample_pair_arr = np.array([f"{x},{y}" for x, y in sample_pairs])

    n_windows = len(windows)
    chrom_arr = np.full(n_windows, args.chromosome_number)
    kwargs = {
        "sample_names": sample_arr,
        "sample_pairs": sample_pair_arr,
        "r_bins": r_bins,
        "chroms": chrom_arr,
        "windows": windows
    }
    if args.parse_site_counts and args.parse_one_locus:
        count_sites()
    if args.parse_site_counts and args.parse_two_locus:
        count_site_pairs()
    if args.parse_one_locus and args.parse_one_sample:
        get_H()
    if args.parse_one_locus and args.parse_two_sample:
        get_Hxy()
    if args.parse_two_locus and args.parse_one_sample:
        get_H2()
    if args.parse_two_locus and args.parse_two_sample:
        get_H2xy()
    np.savez(args.out_fname, **kwargs)
    t = np.round(time.time() - t0, 0)
    time_now = time.strftime("%H:%M:%S", time.localtime())
    print(f"chromosome {args.chromosome_number} parsed in\t{t} s\t@ {time_now}")
