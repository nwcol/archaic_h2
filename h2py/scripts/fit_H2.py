
import argparse

from h2py import inference
from h2py.h2stats_mod import H2stats


def get_args():
    
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-d',
        '--data_file',
        type=str,
        required=True
    )
    parser.add_argument(
        '-g',
        '--graph_file',
        type=str,
        required=True
    )
    parser.add_argument(
        '-p',
        '--param_file',
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
        '--out_file',
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
        '--max_iter',
        type=int,
        default=500
    )
    parser.add_argument(
        '--method',
        type=str,
        default='NelderMead'
    )
    parser.add_argument(
        '--include_H',
        type=int,
        default=0
    )
    return parser.parse_args()


def main():
    """
    
    """
    args = get_args()
    data = inference.load_H2(args.data_file, graph=args.graph_file)
    inference.fit_H2(
        args.graph_file,
        args.param_file,
        data,
        u=args.u,
        perturb=args.perturb,
        verbose=args.verbose,
        method=args.method,
        max_iter=args.max_iter,
        out_file=args.out_file
    )
    return


if __name__ == '__main__':
    main()
    