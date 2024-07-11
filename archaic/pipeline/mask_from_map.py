"""
Make a mask with one region representing map coverage
"""

import argparse
import numpy as np
from archaic import masks
from archaic import two_locus
from archaic import utils


def get_args():
    # get args
    parser = argparse.ArgumentParser()
    parser.add_argument('-r', '--map_fname', required=True)
    parser.add_argument('-o', '--out_fname', required=True)
    parser.add_argument('-n', '--chrom_num', type=int, required=True)
    return parser.parse_args()


def main():
    #
    args = get_args()
    positions, _ = two_locus.read_map_file(args.map_fname)
    start, stop = positions[0], positions[-1]
    regions = np.array([[start, stop + 1]])
    masks.write_regions(regions, args.out_fname, args.chrom_num)
    n_sites = stop + 1 - start
    print(
        utils.get_time(),
        f'map coverage mask for chrom {args.chrom_num} written; {n_sites} sites'
    )
    return 0


if __name__ == '__main__':
    main()
