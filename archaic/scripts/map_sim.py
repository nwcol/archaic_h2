"""
designations


"""
import argparse
import demes
import numpy as np

from archaic import util, parsing, simulation


def get_args():
    #
    parser = argparse.ArgumentParser()
    parser.add_argument('-r', '--rmap', required=True)
    parser.add_argument('-u', '--umap', required=True)
    parser.add_argument('-b', '--mask', required=True)
    parser.add_argument('-g', '--graph')
    parser.add_argument('--windows', required=True)
    parser.add_argument('--tag', default='')
    parser.add_argument('--bins')
    parser.add_argument('--cluster_id', default=None)
    parser.add_argument('--process_id', default=None)
    return parser.parse_args()


def main():
    args = get_args()

    # get the output filename stem
    c = '' if args.cluster_id is None else f'{args.cluster_id}-'
    p = '' if args.process_id is None else f'{args.process_id}'
    chrom_num = args.process_id
    if len(args.tag) > 0:
        tag = f'{args.tag}_{c}{p}'
    else:
        tag = f'{c}{p}'

    bins = np.loadtxt(args.bins)
    windows = np.loadtxt(args.windows)
    graph = demes.load(args.graph)

    # load mask
    regions = util.read_mask_file(args.mask)
    positions = util.get_mask_positions(regions)

    # u setup
    edges, windowed_u = util.read_u_bedgraph(args.umap)
    idx = np.searchsorted(edges[1:], positions)
    mean_u = windowed_u[idx].mean()
    print(f'mean u in mask: {mean_u}')

    # r setup
    rcoords, rvals = util.read_map_file(args.rmap)
    map_length = rvals[-1]
    map_span = rcoords[-1] - rcoords[0]
    mean_r = map_length / map_span / 100
    map_end_val = map_span * mean_r * 100
    # we need to write a dummy map file for parsing. rate column is not needed
    unif_rmap = f'unif-rmap_{chrom_num}.txt'
    with open(unif_rmap, 'w') as file:
        file.write(
            'Position(bp)\tRate(cM/Mb)\tMap(cM)\n'
            f'{int(rcoords[0])}\t0\t0\n'
            f'{int(rcoords[-1])}\t0\t{map_end_val}'
        )

    # simulate!
    sim_fnames = []
    designations = []

    for u_map, u_type in zip([mean_u, args.umap], ['fu', 'eu']):
        for r_map, r_type in zip([mean_r, args.rmap], ['fr', 'er']):
            designation = f'{u_type}-{r_type}'
            designations.append(designation)
            fname = f'{tag}_{designation}.vcf'
            simulation.simulate_chromosome(
                graph,
                fname,
                u=u_map,
                r=r_map,
                L=positions[-1]
            )
            sim_fnames.append(fname)

    # then parse
    for sim_fname, designation in zip(sim_fnames, designations):
        for rmap, r_type in zip([unif_rmap, args.rmap], ['unif', 'emp']):
            dic = parsing.parse_weighted_H2(
                args.mask,
                sim_fname,
                rmap,
                args.umap,
                bins=bins,
                windows=windows
            )
            np.savez(f'{tag}_{designation}_eu_{r_type}.npz', **dic)

            dic = parsing.parse_H2(
                args.mask,
                sim_fname,
                rmap,
                windows=windows,
                bins=bins
            )
            np.savez(f'{tag}_{designation}_fu_{r_type}.npz', **dic)



if __name__ == '__main__':
    main()




