"""
produce a mask representing the intersection of sites in two or more masks
"""
import argparse

import numpy as np

from archaic import util


def get_args():
    # get args
    parser = argparse.ArgumentParser()
    parser.add_argument('-b', '--mask_fnames', nargs='*', required=True)
    parser.add_argument('-s', '--negative_mask_fnames', nargs='*', default=[])
    parser.add_argument('--min_length', type=int, default=None)
    parser.add_argument('-o', '--out_fname', required=True)
    return parser.parse_args()


def main():
    #
    args = get_args()
    masks = [util.read_mask_file(fname) for fname in args.mask_fnames]
    chrom_nums = [utils.read_mask_chrom_num(x) for x in args.mask_fnames]
    if len(np.unique(chrom_nums)) > 1:
        raise ValueError(
            f'you are attempting to intersect masks on chromosomes {chrom_nums}'
        )
    chrom_num = chrom_nums[0]
    intersect = utils.intersect_masks(*masks)
    if len(args.negative_mask_fnames) > 0:
        sub_masks = [
            util.read_mask_file(x) for x in args.negative_mask_fnames
        ]
        sub_union = util.add_masks(*sub_masks)
        intersect = utils.subtract_masks(intersect, sub_union)
    if args.min_length:
        intersect = util.filter_mask_by_length(intersect, args.min_length)
    util.write_mask_file(intersect, args.out_fname, chrom_num)
    n_sites = util.get_bool_mask(intersect).sum()
    print(
        util.get_time(),
        f'intersect mask for chrom {chrom_num} written; {n_sites} sites'
    )
    return 0


if __name__ == '__main__':
    main()
