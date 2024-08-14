"""

"""


import argparse
from archaic import masks
from archaic import utils


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
    chrom_num = masks.check_chroms(args.mask_fnames)
    regs = [masks.read_mask_regions(fname) for fname in args.mask_fnames]
    intersect = masks.intersect_masks(*regs)
    if len(args.negative_mask_fnames) > 0:
        sub_masks = [
            masks.read_mask_regions(x) for x in args.negative_mask_fnames
        ]
        sub_union = masks.add_masks(*sub_masks)
        intersect = masks.subtract_masks(intersect, sub_union)
    if args.min_length:
        intersect = masks.filter_regions_by_length(intersect, args.min_length)
    masks.write_regions(intersect, args.out_fname, chrom_num)
    n_sites = masks.get_n_sites(intersect)
    print(
        utils.get_time(),
        f'intersect mask for chrom {chrom_num} written; {n_sites} sites'
    )
    return 0


if __name__ == '__main__':
    main()
