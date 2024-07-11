"""

"""


import argparse
import gzip
import numpy as np
from archaic import masks
from archaic import one_locus
from archaic import utils


def get_args():
    # get args
    parser = argparse.ArgumentParser()
    parser.add_argument('-v', '--vcf_fname', required=True)
    parser.add_argument('-f', '--fasta_fname', required=True)
    parser.add_argument('-o', '--out_fname', required=True)
    parser.add_argument('-m', '--mask_fname', default=None)
    return parser.parse_args()


def main():
    #
    args = get_args()
    regions = masks.read_mask_regions(args.mask_fname)
    variants = one_locus.Variants(args.vcf_fname, mask_regions=regions)
    positions = variants.positions
    chrom = variants.chrom
    _alleles, fa_header = one_locus.read_fasta_file(args.fasta_fname)
    alleles = _alleles[positions - 1]
    with open(args.out_fname, 'wb') as file:
        header = '#CHROM\tPOS\tINFO/AA\n'.encode()
        file.write(header)
        for i, pos in enumerate(positions):
            line = f'{chrom}\t{pos}\t{alleles[i]}\n'.encode()
            file.write(line)
    print(utils.get_time(), f'annotation saved at {args.out_fname}')
    return 0


if __name__ == "__main__":
    main()
