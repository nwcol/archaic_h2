"""
Each individual is treated as a population
"""


import argparse
from archaic import parsing


def get_args():
    # get args
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--mask_fname', required=True)
    parser.add_argument('-v', '--vcf_fname', required=True)
    parser.add_argument('-f', '--fasta_fname', required=True)
    parser.add_argument('-o', '--out_fname', required=True)
    return parser.parse_args()


def main():
    #
    args = get_args()
    parsing.parse_SFS(
        args.mask_fname,
        args.vcf_fname,
        args.fasta_fname,
        args.out_fname
    )
    return 0


if __name__ == "__main__":
    main()
