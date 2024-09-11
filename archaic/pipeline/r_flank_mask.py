"""
From a mask, remove regions covered in a different mask plus a recombination
distance threshold
"""
import argparse
import numpy as np

from archaic import two_locus, masks, util


def get_args():
    #
    parser = argparse.ArgumentParser()
    parser.add_argument('-r', '--map_fname', required=True)
    parser.add_argument('-b1', '--mask_fname', required=True)
    parser.add_argument('-b2', '--negative_mask_fname', required=True)
    parser.add_argument('--r_thresh', required=True, type=float)
    parser.add_argument('-o', '--out_fname', required=True)
    return parser.parse_args()


def main():
    #
    args = get_args()
    mask1 = masks.Mask.from_bed_file(args.mask_fname)
    n1 = mask1.n_sites
    mask2 = masks.Mask.from_bed_file(args.negative_mask_fname)
    n2 = mask2.n_sites
    max_coord = max(mask1.max(), mask2.max()) + 1
    map_coords = np.arange(0, max_coord)
    map_vals = two_locus.get_r_map(args.map_fname, map_coords)
    cM_thresh = two_locus.map_function(args.r_thresh)
    print(
        util.get_time(),
        f'flank set at {cM_thresh} cM'
    )
    # construct thresh mask
    flank_mask = np.zeros(mask2.shape, dtype=np.int64)
    for i, (start, end) in enumerate(mask2):
        _start, _end = np.searchsorted(
            map_vals, [map_vals[start] - cM_thresh, map_vals[end] + cM_thresh]
        )
        if _end >= max_coord:
            _end = max_coord
        flank_mask[i] = [_start, _end]
    out_mask = masks.subtract_masks(mask1, flank_mask)
    masks.write_regions(out_mask, args.out_fname, mask1.chrom_num)
    n_sites = masks.get_n_sites(out_mask)
    print(
        utils.get_time(),
        f'mask for chrom {mask1.chrom_num} written; '
        f'{n1} sites less {n2} sites yielding {n_sites} sites'
    )


if __name__ == '__main__':
    main()
