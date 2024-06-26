"""
Construct a mask by subtracting a union of negative masks from an 
intersection of positive masks. Optionally remove regions below a given length
"""

import argparse
from archaic import masks


def get_args():
    # get args
    parser = argparse.ArgumentParser()
    parser.add_argument("-pos", "--pos_mask_fnames", required=True, nargs='*')
    parser.add_argument("-neg", "--neg_mask_fnames", default=[], nargs='*')
    parser.add_argument("-o", "--out_fname", required=True)
    parser.add_argument("--min_size", type=int, default=0)
    return parser.parse_args()


def main():
    # 
    args = get_args() 
    chrom_num = check_chroms(args.pos_mask_fnames + args.neg_mask_fnames)
    pos_masks = [masks.read_mask_regions(x) for x in args.pos_mask_names]
    pos_isec = masks.get_mask_intersect(*pos_masks)
    if len(args.neg_mask_fnames) > 0:
        neg_masks = [masks.read_mask_regions(x) for x in args.neg_mask_names]
        neg_union = masks.get_mask_union(*neg_masks)
        pos_isec = masks.subtract_masks(pos_isec, neg_union)
    if args.min_size > 0:
        pos_isec = masks.filter_regions_by_length(pos_isec, args.min_size)
    masks.save_mask_regions(regions, args.out_fname, chrom_num)
    return 0


if __name__ == "__main__":
    main()


