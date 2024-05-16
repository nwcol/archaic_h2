
"""
Parse statistics from a chromosome. Chromosome representations are constructed
from three input types; a .vcf defining variant sites, a .bed mask defining
coverage, and a .txt map file defining the recombination landscape.

Emits a .npz file containing (X, Y are sample names)
    sample_ids
    r_bins
    chroms
    windows



"""

import argparse
import numpy as np
import time
from util import sample_sets
from util import one_locus
from util import two_locus


# names the abbreviations that are used to save statistics
class Abbrevs:

    site_counts = "site_counts"
    site_pair_counts = "site_pair_counts"

    one_locus_one_sample_het_counts = "het_counts"
    one_locus_one_sample_H = "H"

    one_locus_two_sample_het_counts = "two_sample_het_counts"
    one_locus_two_sample_H = "H_xy"

    two_locus_one_sample_het_counts = "het_pair_counts"
    two_locus_one_sample_H = "H2"

    two_locus_two_sample_het_counts = "two_sample_het_pair_counts"
    two_locus_two_sample_H = "H2_xy"


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
        site_counts[i] = \
            np.diff(np.searchsorted(sample_set.positions, window))[0]
    kwargs[Abbrevs.site_counts] = site_counts
    kwargs["n_sites"] = site_counts.sum()


def count_site_pairs():

    site_pair_counts = np.zeros((n_windows, n_bins), dtype=np.int64)

    for i, window in enumerate(windows):
        site_pair_counts[i] = two_locus.count_site_pairs(
            sample_set.positions,
            sample_set.position_map,
            r_bins,
            window=window,
            limit_right=right_lims[i],
            bp_threshold=args.bp_thresh
        )

    kwargs[Abbrevs.site_pair_counts] = site_pair_counts
    kwargs["n_site_pairs"] = site_pair_counts.sum()


def one_locus_one_sample_H():

    counts = np.zeros((n_samples, n_windows), dtype=np.int64)

    for j, sample_id in enumerate(sample_ids):
        sample_counts = np.zeros(n_windows, dtype=np.int64)

        for i, window in enumerate(windows):
            sample_counts[i] = \
                sample_set.count_het_sites(sample_id, window=window)

        counts[j] = sample_counts
        #stat_name = f"{Abbrevs.one_locus_one_sample_het_counts}_{sample_id}"
        #kwargs[stat_name] = sample_counts

        # compute H
        #if args.parse_site_counts:
        #    site_counts = kwargs[Abbrevs.site_counts]
        #    stat_name = f"{Abbrevs.one_locus_one_sample_H}_{sample_id}"
        #    kwargs[stat_name] = sample_counts / site_counts

    kwargs[Abbrevs.one_locus_one_sample_het_counts] = counts

    # compute H
    #if args.parse_site_counts:
    #    site_counts = kwargs[Abbrevs.site_counts]
    #    kwargs[Abbrevs.one_locus_one_sample_H] = all_counts / site_counts


def one_locus_two_sample_H():

    counts = np.zeros((n_sample_pairs, n_windows), dtype=np.float64)

    for j, (sample_x, sample_y) in enumerate(sample_pairs):
        pair_counts = np.zeros(n_windows, dtype=np.float64)

        for i, window in enumerate(windows):
            pair_counts[i] = sample_set.het_xy(sample_x, sample_y, window=window)

        counts[j] = pair_counts
        #stat_name = (f"{Abbrevs.one_locus_two_sample_het_counts}"
         #            f"_{sample_x},{sample_y}")
        #kwargs[stat_name] = pair_counts

        # compute H
        #if args.parse_site_counts:
        #    site_counts = kwargs[Abbrevs.site_counts]
        #    stat_name = (f"{Abbrevs.one_locus_two_sample_H}"
        #                 f"_{sample_x},{sample_y}")
        #    kwargs[stat_name] = pair_counts / site_counts

    kwargs[Abbrevs.one_locus_two_sample_het_counts] = counts

    # compute H
    #if args.parse_site_counts:
    #    site_counts = kwargs[Abbrevs.site_counts]
    #    kwargs[Abbrevs.one_locus_two_sample_H] = all_counts / site_counts


def two_locus_one_sample_H():

    counts = np.zeros((n_samples, n_windows, n_bins), dtype=np.float64)

    for j, sample_id in enumerate(sample_ids):
        sample_counts = np.zeros((n_windows, n_bins), dtype=np.int64)

        for i, window in enumerate(windows):
            sample_counts[i] = two_locus.count_het_pairs(
                sample_set, sample_id, r_bins, window=window,
                limit_right=right_lims[i]
            )
        counts[j] = sample_counts
        #stat_name = f"{Abbrevs.two_locus_one_sample_het_counts}_{sample_id}"
        #kwargs[stat_name] = sample_counts

        # compute H2
        #if args.parse_site_counts:
        #    site_pairs = kwargs[Abbrevs.site_pair_counts]
        #    stat_name = f"{Abbrevs.two_locus_one_sample_H}_{sample_id}"
        #    kwargs[stat_name] = sample_counts / site_pairs

    kwargs[Abbrevs.two_locus_one_sample_het_counts] = counts

    # compute H2
    #if args.parse_site_counts:
    #    site_pairs = kwargs[Abbrevs.site_pair_counts]
    #    kwargs[Abbrevs.two_locus_one_sample_H] = all_counts / site_pairs


def two_locus_two_sample_H():

    counts = np.zeros((n_sample_pairs, n_windows, n_bins), dtype=np.float64)

    for j, (sample_x, sample_y) in enumerate(sample_pairs):
        pair_counts = np.zeros((n_windows, n_bins), dtype=np.float64)

        for i, window in enumerate(windows):
            pair_counts[i] = two_locus.count_two_sample_het_pairs(
                sample_set, sample_x, sample_y, r_bins,  window=window,
                limit_right=right_lims[i]
            )
        counts[j] = pair_counts
        #stat_name = f"{Abbrevs.two_locus_two_sample_het_counts}_{sample_x},{sample_y}"
        #kwargs[stat_name] = pair_counts

        # compute H2
        #if args.parse_site_counts:
        #    site_pairs = kwargs[Abbrevs.site_pair_counts]
        #    stat_name = f"{Abbrevs.two_locus_two_sample_H}_{sample_x},{sample_y}"
        #   kwargs[stat_name] = pair_counts / site_pairs

    kwargs[Abbrevs.two_locus_two_sample_het_counts] = counts

    # compute H2
    #if args.parse_site_counts:
    #    site_pairs = kwargs[Abbrevs.site_pair_counts]
    #    kwargs[Abbrevs.two_locus_two_sample_H] = all_counts / site_pairs


if __name__ == "__main__":
    t0 = time.time()

    parser = argparse.ArgumentParser()
    parser.add_argument("out_fname")
    parser.add_argument("vcf_fname")
    parser.add_argument("mask_fname")
    parser.add_argument("map_fname")

    parser.add_argument("-w", "--window")
    parser.add_argument("-W", "--window_fname", default=None)
    parser.add_argument("-r", "--r_bins")
    parser.add_argument("-R", "--r_bin_fname", default=None)
    parser.add_argument("-s", "--sample_ids", default=None, nargs='*')
    parser.add_argument("-bp", "--bp_thresh", type=int, default=0)

    parser.add_argument("-1l", "--parse_one_locus", type=int, default=1)
    parser.add_argument("-2l", "--parse_two_locus", type=int, default=1)
    parser.add_argument("-c", "--parse_site_counts", type=int, default=1)
    parser.add_argument("-1s", "--parse_one_sample", type=int, default=1)
    parser.add_argument("-2s", "--parse_two_sample", type=int, default=1)
    args = parser.parse_args()

    # interpret optional args
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

    # load up the sample set
    sample_set = sample_sets.SampleSet.read(
        args.vcf_fname, args.mask_fname, args.map_fname
    )
    chrom = sample_set.chrom

    if args.parse_one_sample or args.parse_two_sample:
        if args.sample_ids:
            sample_ids = args.sample_ids
        else:
            sample_ids = sample_set.sample_ids
        sample_arr = np.array(sample_ids)
        n_samples = len(sample_ids)
    else:
        sample_ids = None
        sample_arr = np.array([])
        n_samples = 0

    if args.parse_two_sample:
        sample_pairs = one_locus.enumerate_pairs(sample_ids)
        n_sample_pairs = len(sample_pairs)
        sample_pair_arr = np.array([f"{x},{y}" for x, y in sample_pairs])
    else:
        sample_pairs = None
        sample_pair_arr = np.array([])
        n_sample_pairs = 0

    # set up kwargs to pass to np.savez
    n_windows = len(windows)
    chrom_arr = np.full(n_windows, chrom)
    kwargs = {
        "sample_ids": sample_arr,
        "sample_pairs": sample_pair_arr,
        "r_bins": r_bins,
        "chroms": chrom_arr,
        "windows": windows
    }

    # site counting
    if args.parse_site_counts and args.parse_one_locus:
        count_sites()

    # site pair counting
    if args.parse_site_counts and args.parse_two_locus:
        count_site_pairs()

    # het counting
    if args.parse_one_locus and args.parse_one_sample:
        one_locus_one_sample_H()

    # two sample het counting
    if args.parse_one_locus and args.parse_two_sample:
        one_locus_two_sample_H()

    # two locus het counting
    if args.parse_two_locus and args.parse_one_sample:
        two_locus_one_sample_H()

    # two locus two sample het pair counting
    if args.parse_two_locus and args.parse_two_sample:
        two_locus_two_sample_H()

    np.savez(args.out_fname, **kwargs)
    t = np.round(time.time() - t0, 0)
    time_now = time.strftime("%H:%M:%S", time.localtime())
    print(f"chromosome {chrom} parsed in\t{t} s\t@ {time_now}")
