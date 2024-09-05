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
    parser.add_argument("-s", "--weight_fname", required=True)
    parser.add_argument("-o", "--out_fname", required=True)
    parser.add_argument('--bins', default=None)
    parser.add_argument('-w', '--window_fname', default=None)
    return parser.parse_args()


def main():
    #
    args = get_args()
    if args.bins is not None:
        bins = np.loadtxt(args.bins)
    else:
        bins = None
    if args.window_fname is not None:
        window_arr = np.loadtxt(args.window_fname)
        if window_arr.ndim == 1:
            window_arr = window_arr[np.newaxis]
        windows = window_arr[:, :2]
        bounds = window_arr[:, 2]
    else:
        windows = None
        bounds = None
    dic = parsing.parse_weighted_H2(
        args.mask_fname,
        args.vcf_fname,
        args.map_fname,
        args.weight_fname,
        bins,
        windows=windows,
        bounds=bounds
    )
    np.savez(args.out_fname, **dic)
    return 0


if __name__ == "__main__":
    main()
