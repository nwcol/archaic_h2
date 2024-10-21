
import argparse
import demes
import numpy as np

from h2py import inference
from h2py.h2stats_mod import H2stats


def get_args():
    
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-d',
        '--data_fname',
        type=str,
        required=True
    )
    parser.add_argument(
        '-g',
        '--graph_fname',
        type=str,
        required=True
    )
    parser.add_argument(
        '-p',
        '--param_fname',
        type=str,
        required=True
    )
    parser.add_argument(
        '-u',
        type=float,
        required=True
    )
    parser.add_argument(
        '-o',
        '--out_fname',
        type=str,
        required=True
    )
    # optional args
    parser.add_argument(
        '--perturb',
        type=float,
        default=0
    )
    parser.add_argument(
        '--verbose',
        type=int,
        default=1
    )
    parser.add_argument(
        '--method',
        type=str,
        default='NealderMead'
    )
    return parser.parse_args()


def main():
    """
    
    """
    args = get_args()
    data = H2stats.from_file(args.data_fname, graph=args.graph_fname)
    inference.fit_H2(
        args.graph_fname,
        args.param_fname,
        data,
        u=args.u,
        perturb=args.perturb,
        verbose=args.verbose,
        out_fname=args.out_fname
    )
    return


if __name__ == '__main__':
    main()
    