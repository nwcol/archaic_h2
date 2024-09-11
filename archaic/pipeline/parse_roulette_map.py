"""
parse mutation rates from .vcf
"""
import argparse
import gzip
import numpy as np

from archaic import util, masks


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-v', '--vcf_fname', required=True)
    parser.add_argument('-o', '--out_fname', required=True)
    parser.add_argument('--out_mask_fname', default=None)
    return parser.parse_args()


def main(vcf_fname, out_fname, out_mask_fname=None):
    #
    positions, rates = utils.read_vcf_rate_file(vcf_fname)
    np.savez(out_fname, positions=positions, rates=rates)
    if out_mask_fname:
        chrom = util.read_vcf_file_chrom(vcf_fname)
        mask = masks.Mask.from_positions(positions)
        masks.write_regions(mask, out_mask_fname, chrom)
    print(util.get_time(), f'saved parsed data')
    return 0


if __name__ == '__main__':
    args = get_args()
    main(args.vcf_fname, args.out_fname, out_mask_fname=args.out_mask_fname)
