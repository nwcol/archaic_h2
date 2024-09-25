"""
build a .bed mask file by consecutively
    (1) taking an intersection of sites in the input files flagged -i
    (2) taking a union of sites in input files -s 
    (3) if flank arguments are provided, extending this union with flanking
        regions as specified
    (4) removing all positions in the now-flanked union
    (5) if min_length is provided, remove all regions smaller than it
"""
import argparse
import numpy as np

from archaic import util


def get_args():
    # get args
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--isec_masks', nargs='*', required=True)
    parser.add_argument('-s', '--subtract_masks', nargs='*', default=[])
    parser.add_argument('--flank', type=float, default=None)
    parser.add_argument('--flank_unit', default='bp')
    parser.add_argument('--rmap', default=None)
    parser.add_argument('--min_length', type=int, default=None)
    parser.add_argument('-o', '--out_fname', required=True)
    return parser.parse_args()


def get_n_sites(mask):
    #
    return np.sum(np.diff(mask, axis=1))


def main():
    #
    args = get_args()

    masks = [util.read_mask_file(fname) for fname in args.isec_masks]
    chrom_nums = [util.read_mask_chrom_num(x) for x in args.isec_masks]

    if len(np.unique(chrom_nums)) > 1:
        raise ValueError(
            'you are attempting to intersect masks on multiple chromosomes '
            f'{chrom_nums}'
        )
    chrom_num = chrom_nums[0]

    print(
        util.get_time(),
        f'intersecting {len(masks)} masks with '
        f'{[len(m) for m in masks]} regions and '
        f'{[get_n_sites(m) for m in masks]} sites'
    )

    isec = util.intersect_masks(*masks)

    print(
        util.get_time(),
        f'intersected mask holds {get_n_sites(isec)} sites '
        f'in {len(isec)} regions'
    )

    if len(args.subtract_masks) > 0:
        sub_masks = [util.read_mask_file(x) for x in args.subtract_masks]

        print(
            util.get_time(),
            f'taking union of {len(sub_masks)} masks with '
            f'{[len(m) for m in sub_masks]} regions and '
            f'{[get_n_sites(m) for m in sub_masks]} sites'
        )

        sub_union = util.add_masks(*sub_masks)
        n_union_sites = get_n_sites(sub_union)

        print(
            util.get_time(),
            f'union mask holds {get_n_sites(sub_union)} sites '
            f'in {len(sub_union)} regions'
        )

        if args.flank is not None:
            if args.flank_unit not in ['bp', 'cM']:
                raise ValueError(
                    f'{args.flank_unit} is not a valid unit'
                )

            if args.flank_unit == 'bp':
                sub_union = util.add_mask_flank(sub_union, int(args.flank))

            elif args.flank_unit == 'cM':
                rcoords, rvals = util.read_map_file(args.rmap)
                sub_union = util.add_mask_flank_cM(
                    sub_union, rcoords, rvals, float(args.flank)
                )
            print(
                util.get_time(),
                f'flanking operation added '
                f'{get_n_sites(sub_union) - n_union_sites} to union mask'
            )

        isec = util.subtract_masks(isec, sub_union)

        print(
            util.get_time(),
            f'subtracted mask holds {get_n_sites(isec)} sites '
            f'in {len(isec)} regions'
        )

    if args.min_length is not None:
        if args.min_length > 0:
            isec = util.filter_mask_by_length(isec, int(args.min_length))

            print(
                util.get_time(),
                f'length-thresholded mask holds {get_n_sites(isec)} sites '
                f'in {len(isec)} regions'
            )

    util.write_mask_file(isec, args.out_fname, chrom_num)

    n_sites = util.get_bool_mask(isec).sum()
    print(
        util.get_time(),
        f'isec mask for chrom {chrom_num} written with {n_sites} sites'
    )
    return 0


if __name__ == '__main__':
    main()
