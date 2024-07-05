"""

"""


import argparse
from archaic import masks


def get_args():
    # get args
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--mask_fnames', nargs='*', required=True)
    parser.add_argument('-o', '--out_fname', required=True)
    return parser.parse_args()


def main():

    args = get_args()
    chrom_num = masks.check_chroms(args.mask_fnames)
    regs = [masks.read_mask_regions(fname) for fname in args.mask_fnames]
    isec = masks.intersect_masks(*regs)
    masks.write_regions(isec, args.out_fname, chrom_num)
    return 0


if __name__ == '__main__':
    main()
