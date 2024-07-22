"""
Get and print standard errors about inferred graph parameters
"""
import argparse
import demes
import numpy as np

from archaic import inference
from archaic.spectra import H2Spectrum


def get_args():
    # get args
    parser = argparse.ArgumentParser()
    parser.add_argument('-g', '--graph_fname', required=True)
    parser.add_argument('-p', '--options_fname', required=True)
    parser.add_argument('-d', '--data_fname', required=True)
    parser.add_argument('-u', '--u', type=float, default=1.35e-8)
    parser.add_argument('--delta', type=float, default=0.01)
    parser.add_argument('-n', '--n_bootstraps', type=int, default=None)
    parser.add_argument('--method', default='GIM')
    return parser.parse_args()


def main():
    #
    args = get_args()
    graph = demes.load(args.graph_fname)
    data = H2Spectrum.from_bootstrap_file(args.data_fname, graph=graph)
    if args.method == 'GIM':
        if args.n_bootstraps:
            n = args.n_bootstraps
        else:
            file = np.load(args.data_fname)
            n = len(file['H2_dist'])
        bootstraps = [
            H2Spectrum.from_bootstrap_distribution(
                args.data_fname, i, sample_ids=data.sample_ids
            ) for i in range(n)
        ]
    else:
        bootstraps = None
    pnames, p0, std_errs = inference.get_uncerts(
        args.graph_fname,
        args.options_fname,
        data,
        bootstraps=bootstraps,
        u=args.u,
        delta=args.delta,
        method=args.method
    )
    print('param\tfit\tstderr')
    for name, p, s in zip(pnames, p0, std_errs):
        if p > 1:
            p = np.round(p, 1)
        else:
            p = np.format_float_scientific(p, precision=2)
        if s > 1:
            s = np.round(s, 1)
        else:
            s = np.format_float_scientific(s, precision=2)
        print(f'{name}\t{p}\t{s}')


if __name__ == "__main__":
    main()
