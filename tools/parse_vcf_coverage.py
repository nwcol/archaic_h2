"""
Write a .bed file recording the positions covered in a .vcf file
"""
import argparse
import numpy as np

from h2py import util


def get_args():
    # get args
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-v', '--vcf_file',
        type=str, 
        required=True
    )
    parser.add_argument(
        '-o', '--out_file',
        type=str,
        required=True
    )
    return parser.parse_args()


def main():
    # get positions and convert to regions; save
    args = get_args() 

    print(util.get_time(), f'parsing coverage from {args.vcf_file}')

    chrom_num, positions = util.read_vcf_positions(args.vcf_file)
    mask = np.ones(len(positions))
    mask[positions] = 0
    regions = util.mask_to_regions(mask)
    util.write_bedfile(args.out_file, chrom_num, regions=regions, header=False)

    print(util.get_time(), f'coverage written at {args.out_file}')
    return 0


if __name__ == "__main__":
    main()

