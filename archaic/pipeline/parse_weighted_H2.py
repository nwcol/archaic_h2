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
    parser.add_argument('-w', '--windows', default=None)
    return parser.parse_args()


def main():
    #
    args = get_args()
    dic = parsing.parse_weighted_H2(
        args.mask_fname,
        args.vcf_fname,
        args.map_fname,
        args.weight_fname,
        bins=args.bins,
        windows=args.windows
    )
    np.savez(args.out_fname, **dic)
    return 0


if __name__ == "__main__":
    main()
