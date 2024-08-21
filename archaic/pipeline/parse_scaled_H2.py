"""

"""
import argparse
import numpy as np

from archaic import parsing


def get_args():
    # get args
    parser = argparse.ArgumentParser()
    parser.add_argument("-v", "--vcf_fname", required=True)
    parser.add_argument("-b", "--mask_fname", required=True)
    parser.add_argument("-r", "--map_fname", required=True)
    parser.add_argument("-u", "--u_fname", required=True)
    parser.add_argument("-o", "--out_fname", required=True)
    parser.add_argument("-B", "--r_bin_fname", default=None)
    parser.add_argument('-W', '--window_fname')
    return parser.parse_args()


def main():
    #
    args = get_args()
    r_bins = np.loadtxt(args.r_bin_fname)
    window_arr = np.loadtxt(args.window_fname)
    windows = window_arr[:, :2]
    bounds = window_arr[:, 2]
    dic = parsing.parse_scaled_H2(
        args.mask_fname,
        args.vcf_fname,
        args.map_fname,
        args.u_fname,
        r_bins,
        windows=windows,
        bounds=bounds
    )
    np.savez(args.out_fname, **dic)
    return 0


if __name__ == "__main__":
    main()
