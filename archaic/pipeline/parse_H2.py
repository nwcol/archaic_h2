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
    parser.add_argument("-o", "--out_fname", required=True)
    parser.add_argument("-w", "--window_fname", default=None)
    parser.add_argument('--get_two_sample', type=int, default=1)
    parser.add_argument('--bins', default=None)
    return parser.parse_args()


def main():
    #
    args = get_args()

    if args.window_fname is not None:
        try:
            arr = np.loadtxt(args.window_fname)
        except:
            arr = np.array(eval(args.window_fname))
        if arr.ndim == 1:
            arr = arr[np.newaxis]
        windows = arr[:, :2]
        bounds = arr[:, 2]
    else:
        windows = None
        bounds = None

    if args.bins is not None:
        try:
            bins = np.loadtxt(args.bins)
        except:
            bins = np.array(eval(args.bins))
    else:
        bins = None

    dic = parsing.parse_H2(
        args.mask_fname,
        args.vcf_fname,
        args.map_fname,
        windows=windows,
        bounds=bounds,
        bins=bins,
        get_two_sample=args.get_two_sample
    )
    np.savez(args.out_fname, **dic)
    return 0


if __name__ == "__main__":
    main()
