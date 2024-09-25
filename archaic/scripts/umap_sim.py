
import argparse
import demes
import numpy as np
import sys

from archaic import util, parsing, simulation


def get_args():
    #
    parser = argparse.ArgumentParser()
    parser.add_argument('-g', required=True)
    parser.add_argument('-L', type=int, required=True)
    parser.add_argument('-u', required=True)
    parser.add_argument('-r', required=True)
    parser.add_argument('--mask_fname', required=True)
    parser.add_argument('--windows', required=True)
    parser.add_argument('--tag', default='')
    parser.add_argument('--bins')
    parser.add_argument('--cluster_id', default=None)
    parser.add_argument('--process_id', default=None)
    return parser.parse_args()


def main():

    args = get_args()
    c = '' if args.cluster_id is None else f'{args.cluster_id}-'
    p = '' if args.process_id is None else f'{args.process_id}'
    if len(args.tag) > 0:
        tag = f'{args.tag}_{c}{p}_'
    else:
        tag = f'{c}{p}_'

    bins = np.loadtxt(args.bins)
    windows = np.loadtxt(args.windows)
    graph = demes.load(args.g)

    # measure mean r, u in the empirical maps
    regions = util.read_mask_file(args.mask_fname)
    positions = util.get_mask_positions(regions)

    edges, windowed_u = util.read_u_bedgraph(args.u)
    idx = np.searchsorted(edges[1:], positions)
    mean_u = windowed_u[idx].mean()
    print(f'mean u in mask: {mean_u}')

    # simulate with empirical u-map and parse the simulated data
    emp_vcf = f'{tag}emp_u.vcf'
    simulation.simulate_chromosome(
        graph,
        emp_vcf,
        u=args.u,
        r=args.r,
        L=args.L
    )
    unif_vcf = f'{tag}flat_u.vcf'
    simulation.simulate_chromosome(
        graph,
        unif_vcf,
        u=mean_u,
        r=args.r,
        L=args.L
    )
    for name, vcf_fname in zip(['empirical_u', 'unif_u'], [emp_vcf, unif_vcf]):
        dic = parsing.parse_weighted_H2(
            args.mask_fname,
            vcf_fname,
            args.r,
            args.u,
            bins=bins,
            windows=windows
        )
        np.savez(f'{tag}{name}-weighted.npz', **dic)
        dic = parsing.parse_H2(
            args.mask_fname,
            vcf_fname,
            args.r,
            windows=windows,
            bins=bins
        )
        np.savez(f'{tag}{name}-flat.npz', **dic)


if __name__ == '__main__':
    main()

