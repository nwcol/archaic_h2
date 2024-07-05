"""

"""


import argparse
from archaic import parsing


def get_args():
    # get args
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--in_fnames", nargs='*', required=True)
    parser.add_argument("-o", "--out_fname", required=True)
    parser.add_argument("-n", "--n_iters", type=int, default=1_000)
    return parser.parse_args()


def main():
    # call the bootstrap function in the parsing module
    args = get_args()
    parsing.bootstrap_H2(args.in_fnames, args.out_fname, args.n_iters)
    return 0


if __name__ == "__main__":
    main()
