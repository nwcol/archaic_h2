
import argparse
import numpy as np
import pickle

from h2py import h2_parsing


def get_args():

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-i', '--in_files',
        nargs='*',
        type=str,
        required=True
    )
    parser.add_argument(
        '-o', '--out_file',
        type=str,
        required=True
    )
    parser.add_argument(
        '--num_reps',
        type=int,
        default=None
    )
    parser.add_argument(
        '--num_samples',
        type=int,
        default=None
    )
    parser.add_argument(
        '--mut_weighted',
        type=int,
        default=0
    )
    parser.add_argument(
        '--mean_mut',
        type=float,
        default=None
    )
    return parser.parse_args()


def main():
    """
    
    """
    args = get_args()

    regions = {}
    for in_file in args.in_files:
        with open(in_file, 'rb') as fin:
            file_regs = pickle.load(fin)
        for region in file_regs:
            if region in regions:
                raise ValueError(f'{region} occurs twice in input')
            else:
                regions[region] = file_regs[region]

    bootstrapped_stats = h2_parsing.bootstrap_H2(
        regions, 
        num_reps=args.num_reps, 
        num_samples=args.num_samples,
        mut_weighted=args.mut_weighted,
        to_mean_mut=args.mean_mut
    )
    out = {'bootstrap': bootstrapped_stats}

    with open(args.out_file, 'wb') as fout:
        pickle.dump(out, fout)

    return 


if __name__ == '__main__':
    main()

