
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
    parser.add_argument('--cluster_id', default='0')
    parser.add_argument('--process_id', default='0')
    parser.add_argument('--tag', default='')
    parser.add_argument('--bins')
    return parser.parse_args()


def main():

    args = get_args()

    tag = f'{args.tag}_{args.cluster}_{args.process}'

    bins = np.loadtxt(args.bins)
    win_arr = np.loadtxt('blocks_1.txt')
    windows = win_arr[:, :2]
    bounds = win_arr[:, 2]
    cent_bounds = np.full(len(bounds), bounds[-1])

    graph = demes.load(args.graph_fname)

    # measure mean r, u in the empirical maps
    regions = util.read_mask_file(args.mask_fname)
    positions = util.get_mask_positions(regions)

    edges, windowed_u = util.read_u_bedgraph(args.u)
    idx = np.searchsorted(edges[1:], positions)
    mean_u = windowed_u[idx].mean()
    print(f'mean u in mask: {mean_u}')

    # simulate with empirical u-map and parse the simulated data
    emp_vcf = f'{tag}_emp_u.vcf'
    simulation.simulate_chromosome(
        graph,
        emp_vcf,
        u=args.u,
        r=args.r,
        L=args.L
    )
    #flat_vcf = f'{tag}_flat_u.vcf'
    #simulation.simulate_chromosome(
    #    graph,
    #    flat_vcf,
    #    u=mean_u,
    #    r=args.r,
    #    L=args.L
    #)

    for name, _bounds in zip(['bounded', 'not_bounded'], [bounds, cent_bounds]):
        dic1 = parsing.parse_weighted_H2(
            args.mask_fname,
            emp_vcf,
            args.r,
            args.u,
            bins=bins,
            windows=windows,
            bounds=_bounds
        )
        np.savez(f'{tag}-weighted-{_bounds}.npz', **dic1)
        dic2 = parsing.parse_H2(
            args.mask_fname,
            emp_vcf,
            args.r,
            windows=windows,
            bounds=_bounds,
            bins=bins,
        )
        np.savez(f'{tag}-flat-{bounds}.npz', **dic2)


if __name__ == '__main__':
    #main()
    print('yippee')
