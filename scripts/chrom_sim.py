
import argparse
import demes
import numpy as np
import sys

from archaic import utils, parsing, simulation


def get_args():
    #
    parser = argparse.ArgumentParser()
    parser.add_argument('-g', required=True)
    parser.add_argument('-L', type=int, required=True)
    parser.add_argument('-u', type=float, required=True)
    parser.add_argument('-r', type=float, required=True)
    parser.add_argument('--window_size', type=int, default=10000000)
    parser.add_argument('--centromere', type=int, default=None)
    parser.add_argument('--cluster_id', default='0')
    parser.add_argument('--process_id', default='0')
    parser.add_argument('--tag', default='')
    return parser.parse_args()


def main():
    #
    args = get_args()

    bins = np.concatenate(
        ([0], np.logspace(-6, -1, 21)[:-1], np.logspace(-1, -0.30189, 10))
    )
    L = int(args.L)
    w = int(args.window_size)
    window_edges = np.arange(0, L + w, w)
    windows = np.zeros((len(window_edges) - 1, 2), dtype=int)
    windows[:, 0] = window_edges[:-1]
    windows[:, 1] = window_edges[1:]
    bounds = np.full(len(windows), window_edges[-1], dtype=int)
    if args.centromere is not None:
        bounds[windows[:, 1] <= args.centromere] = args.centromere

    graph = demes.load(args.g)

    mask_fname = 'mask.bed'
    with open(mask_fname, 'w') as file:
        file.write(
            f'chr0\t0\t{L}'
        )
    rmap_fname = 'rmap.txt'
    r = float(args.r)
    map_span = L * r * 100
    with open(rmap_fname, 'w') as file:
        file.write(
            'Position(bp)\tMap(cM)\n'
            f'0\t{0}\n'
            f'{L}\t{map_span}'
        )
    u = float(args.u)
    vcf_fname = f'{args.tag}_{args.cluster_id}_{args.process_id}.vcf'
    simulation.simulate_chromosome(
        graph,
        vcf_fname,
        u=u,
        r=r,
        contig_id='0',
        L=L
    )
    stat_fname = f'H2_{args.tag}_{args.cluster_id}_{args.process_id}.npz'
    dic = parsing.parse_H2(
        mask_fname,
        vcf_fname,
        rmap_fname,
        windows=windows,
        bounds=bounds,
        bins=bins,
    )
    np.savez(stat_fname, **dic)
    print(f'parsing complete')


if __name__ == '__main__':
    main()
