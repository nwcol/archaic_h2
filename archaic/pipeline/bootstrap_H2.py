"""
Bootstrap some H2 archive files
"""


import argparse
import numpy as np
from archaic import parsing


def get_args():
    # get args
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--in_fnames", nargs='*', required=True)
    parser.add_argument("-o", "--out_fname", required=True)
    parser.add_argument("-n", "--n_iters", type=int, default=1_000)
    parser.add_argument('--name_map', nargs='*', default=[])
    parser.add_argument('--bin_slice', default=None)
    # not actually a slice. inclusive interval like 1-19
    return parser.parse_args()


def main():
    # call the bootstrap function in the parsing module
    args = get_args()
    if args.bin_slice is None:
        bin_slice = None
    else:
        start, end = args.bin_slice.split('-')
        bin_slice = (int(start), int(end))
    files = [np.load(fname) for fname in args.in_fnames]
    dic = parsing.bootstrap_H2(
        files, n_iters=args.n_iters, bin_slice=bin_slice
    )
    if len(args.name_map) > 0:
        ids = dic['ids']
        for mapping in args.name_map:
            old, new = mapping.split(':')
            ids[ids == old] = new
        dic['ids'] = ids
    np.savez(args.out_fname, **dic)
    return 0


if __name__ == "__main__":
    main()
