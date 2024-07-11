"""
Write a mask file recording the positions covered in a .vcf file
"""


import argparse
from archaic import masks
from archaic import utils


def get_args():
    # get args
    parser = argparse.ArgumentParser()
    parser.add_argument('-v', '--vcf_fname', required=True)
    parser.add_argument('-o', '--out_fname', required=True)
    return parser.parse_args()


def main():
    # get positions and convert to regions; save
    args = get_args() 
    print(utils.get_time(), f'parsing mask from {args.vcf_fname}')
    positions, chrom_num = masks.read_vcf_positions(args.vcf_fname)
    regions = masks.positions_to_regions(positions)
    masks.write_regions(regions, args.out_fname, chrom_num)
    print(utils.get_time(), f'mask written at {args.out_fname}')
    return 0


if __name__ == "__main__":
    main()

