"""
Given a numpy array of windows and a genetic map, get a genetic map which is
uniform on each window
"""
import argparse
import numpy as np

from archaic import two_locus


def get_args():
    # get args
    parser = argparse.ArgumentParser()
    parser.add_argument('-r', '--map_fname', required=True)
    parser.add_argument('-w', '--window_fname', required=True)
    parser.add_argument('-o', '--out_fname', required=True)
    return parser.parse_args()


def main():
    #
    args = get_args()
    windows = np.loadtxt(args.window_fname)
    coords = np.hstack([windows[0, 0], windows[:, 1]])
    map_positions = two_locus.get_r_map(args.map_fname, coords)
    two_locus.write_map_file(args.out_fname, coords, map_positions)
    return 0


if __name__ == '__main__':
    main()
