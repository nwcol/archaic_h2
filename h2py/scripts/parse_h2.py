
import argparse
import numpy as np
import pickle

from h2py import util, h2_parsing


def get_args():

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-v', '--vcf_file',
        type=str,
        required=True
    )
    parser.add_argument(
        '-b', '--bed_file',
        type=str
    )
    parser.add_argument(
        '-r', '--rec_map_file',
        type=str,
        required=True
    )
    parser.add_argument(
        '-m', '--mut_map_file',
        type=str
    )
    parser.add_argument(
        '-p', '--pop_file',
        type=str
    )
    parser.add_argument(
        '-R', '--region_file',
        type=str
    )
    parser.add_argument(
        '-o', '--out_file',
        type=str,
        required=True
    )
    parser.add_argument(
        '-bins', '--bin_file',
        type=str,
        required=True
    )
    parser.add_argument(
        '--min_reg_len',
        type=int
    )
    parser.add_argument(
        '--compute_denom',
        type=int,
        default=1
    )
    parser.add_argument(
        '--compute_snp_denom',
        type=int,
        default=1
    )
    parser.add_argument(
        '--compute_two_sample',
        type=int,
        default=1
    )
    parser.add_argument(
        '--chrom',
        type=str,
        default=1
    )
    parser.add_argument(
        '--verbose',
        type=str,
        default=1
    )
    return parser.parse_args()


def main():
    """
    
    """
    args = get_args()

    bins = np.loadtxt(args.bin_file)

    if args.region_file is not None:
        regions = np.loadtxt(args.region_file, usecols=(0,1,2))
        if regions.ndim == 1:
            regions = regions[np.newaxis]
    else:
        regions = [None]

    if args.verbose:
        print(util.get_time(), f'computing H2 on chromosome {args.chrom}')

    region_stats = {}
    for i, region in enumerate(regions):
        key = f'{args.chrom}_{i}'
        region_stats[key] = h2_parsing.compute_H2(
            args.vcf_file,
            bed_file=args.bed_file,
            rec_map_file=args.rec_map_file,
            mut_map_file=args.mut_map_file,
            pop_file=args.pop_file,
            region=region,
            r_bins=bins,
            phased=False,
            min_reg_len=args.min_reg_len,
            compute_denom=args.compute_denom,
            compute_snp_denom=args.compute_snp_denom,
            compute_two_sample=args.compute_two_sample,
            verbose=args.verbose
        )

    with open(args.out_file, 'wb') as fout:
        pickle.dump(region_stats, fout)

    return 


if __name__ == '__main__':
    main()

