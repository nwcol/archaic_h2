"""

"""


import argparse
import numpy as np
from archaic import parsing


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


def main():
    #
    args = get_args()
    if args.window:
        windows = np.array(eval(args.window))
    elif args.window_fname:
        windows = np.loadtxt(args.window_fname)
    else:
        print("using default single window")
        windows = None
    if windows.ndim != 2:
        raise ValueError(f"windows must be dim2, but are dim{windows.ndim}")
    if windows.shape[1] == 3:
        bounds = windows[:, 2]
        windows = windows[:, :2]
    else:
        bounds = None
    if args.r_bins:
        r_bins = np.array(eval(args.r_bins))
    elif args.r_bin_fname:
        r_bins = np.loadtxt(args.r_bin_fname)
    else:
        r_bins = np.logspace(-6, -2, 17)
        print(f'using default r bins {r_bins}')
    dic = parsing.parse_H2(
        args.mask_fname,
        args.vcf_fname,
        args.map_fname,
        windows,
        bounds,
        r_bins
    )
    np.savez(args.out_fname, **dic)
    return 0


if __name__ == "__main__":
    main()
