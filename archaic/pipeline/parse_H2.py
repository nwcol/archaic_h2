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
    parser.add_argument("-r", "--rmap_fname", required=True)
    parser.add_argument("-o", "--out_fname", required=True)
    parser.add_argument("-w", "--windows", default=None)
    parser.add_argument('--bins', default=None)
    parser.add_argument('--get_two_sample', type=int, default=1)
    return parser.parse_args()


def main():
    #
    args = get_args()
    dic = parsing.parse_H2(
        args.mask_fname,
        args.vcf_fname,
        args.rmap_fname,
        windows=args.windows,
        bins=args.bins,
        get_two_sample=args.get_two_sample
    )
    np.savez(args.out_fname, **dic)
    return 0


if __name__ == "__main__":
    main()
