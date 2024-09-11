"""

"""
import argparse

from archaic import masks, util


def get_args():
    # get args
    parser = argparse.ArgumentParser()
    parser.add_argument('-b1', '--mask_fname', required=True)
    parser.add_argument('-b2', '--negative_mask_fname', default=[])
    parser.add_argument('-o', '--out_fname', required=True)
    return parser.parse_args()


def main():
    #
    args = get_args()
    mask1 = masks.Mask.from_bed_file(args.mask_fname)
    n1 = mask1.n_sites
    mask2 = masks.Mask.from_bed_file(args.negative_mask_fname)
    n2 = mask2.n_sites
    out_mask = masks.subtract_masks(mask1, mask2)
    masks.write_regions(out_mask, args.out_fname, mask1.chrom_num)
    n_sites = masks.get_n_sites(out_mask)
    print(
        utils.get_time(),
        f'mask for chrom {mask1.chrom_num} written; '
        f'{n1} sites less {n2} sites yielding {n_sites} sites'
    )
    return 0


if __name__ == '__main__':
    main()
