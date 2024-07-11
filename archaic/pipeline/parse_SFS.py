"""
Each individual is treated as a population. Parses an arbitrary number of masks
and .vcf files. .vcf files must have 'AA' field in INFO
"""


import argparse
from archaic import parsing


def get_args():
    # get args
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--mask_fnames', nargs='*', required=True)
    parser.add_argument('-v', '--vcf_fnames', nargs='*', required=True)
    parser.add_argument('-o', '--out_fname', required=True)
    parser.add_argument('--ref_as_ancestral', type=int, default=0)
    return parser.parse_args()


def main():
    #
    args = get_args()
    parsing.parse_SFS(
        args.mask_fnames,
        args.vcf_fnames,
        args.out_fname,
        ref_as_ancestral=args.ref_as_ancestral
    )
    return 0


if __name__ == "__main__":
    main()
