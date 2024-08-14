"""
estimate average covariance between SNPs in bins of r
"""
import argparse
import numpy as np

from archaic import utils


def get_args():
    #
    parser = argparse.ArgumentParser()
    parser.add_argument('-v', '--vcf_fname', required=True)
    parser.add_argument('-r', '--map_fname', required=True)
    parser.add_argument('-b', '--mask_fname', default=None)
    parser.add_argument('-o', '--out_fname', required=True)
    return parser.parse_args()


def compute_cov(genotypes, r_map, bins):
    # not vectorized


    return 0


def main():
    #
    args = get_args()
    if args.mask_fname is not None:
        _, mask_positions = utils.read_mask_file(args.mask_fname)
        print(utils.get_time(), 'read mask file')
    else:
        mask_positions = None
    sample_ids, vcf_positions, genotype_arr = \
        utils.read_vcf_file(args.vcf_fname, mask_positions=mask_positions)
    print(utils.get_time(), 'read vcf file')
    vcf_r_map = utils.read_map_file(args.map_fname, vcf_positions)
    print(utils.get_time(), 'read map file')

    bins = np.linspace(-6, -2, 17)

    for i in range(len(sample_ids)):
        compute_cov(genotype_arr, vcf_r_map, bins)


    return 0


if __name__ == '__main__':
    pass

