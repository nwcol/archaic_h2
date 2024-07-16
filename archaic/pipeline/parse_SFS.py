"""
Each individual is treated as a population. Parses an arbitrary number of masks
and .vcf files. .vcf files must have 'AA' field in INFO
"""


import argparse
from archaic import parsing
from archaic import utils


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
    # a hack. works when all .vcfs have the samek mask length though
    if len(args.mask_fnames) == 1 and len(args.vcf_fnames) > 1:
        mask_fnames = [args.mask_fnames[0]] * len(args.vcf_fnames)
        print(utils.get_time(), "using same mask for all .vcfs...")
    else:
        mask_fnames = args.mask_fnames
    parsing.parse_SFS(
        mask_fnames,
        args.vcf_fnames,
        args.out_fname,
        ref_as_ancestral=args.ref_as_ancestral
    )
    return 0


if __name__ == "__main__":
    main()
