
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
    site_counts = one_locus.parse_site_counts(mask_pos, windows)
    H_counts = one_locus.parse_H_counts(genotypes, vcf_pos, windows)
    pair_counts = two_locus.parse_pair_counts(
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
        site_counts=site_counts,
        H_counts=H_counts,
        pair_counts=pair_counts,
        H2_counts=H2_counts
    )
    np.savez(out_fname, **arrs)
    t = np.round(time.time() - t0, 0)
    print(utils.get_time(), f"chromosome parsed in\t{t} s")


def main():
    #
    args = get_args()
    if args.window:
        windows = np.array(eval(args.window))
    elif args.window_fname:
        windows = np.loadtxt(args.window_fname)
    else:
        raise ValueError("you must provide windows!")
    if windows.ndim != 2:
        raise ValueError(f"windows must be dim2, but are dim{windows.ndim}")
    ###
    if windows.shape[1] == 3:
        bounds = windows[:, 2]
        windows = windows[:, :2]
    else:
        bounds = np.repeat(windows[-1, 1], len(windows))
    ###
    if args.r_bins:
        r_bins = np.array(eval(args.r_bins))
    elif args.r_bin_fname:
        r_bins = np.loadtxt(args.r_bin_fname)
    else:
        raise ValueError("you must provide r bins!")
    parse(
        args.mask_fname,
        args.vcf_fname,
        args.map_fname,
        windows,
        bounds,
        r_bins,
        args.out_fname,
    )
    return 0


if __name__ == "__main__":
    main()
