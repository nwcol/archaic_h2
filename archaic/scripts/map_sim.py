
import argparse
import demes
import msprime
import numpy as np

from archaic import util, parsing


# these define the simulation/parsing configuration
sim_configs = [
    ('unif-u', 'unif-r'),
    ('emp-u', 'emp-r')
]
parse_configs = [
    ('unif-u', 'unif-r'),
    ('emp-u', 'emp-r')
]


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
    parser.add_argument('--count_site_pairs', type=int, default=0)
    parser.add_argument('--dtwf_time', type=int, default=100)
    return parser.parse_args()



def simulate(
    graph,
    out_fname,
    u=None,
    r=None,
    L=None,
    sampled_demes=None,
    contig_id=None,
    dtwf_time=None
):
    # L 'stretches' both maps
    # u, r can be floats or .bedgraph/.txt files holding rates

    def increment1(x):
        # add 1 to each position so that output .vcf is 1-indexed
        return [_ + 1 for _ in x]

    def truncate_map(xs, vals, L):
        # extend or truncate map defined by xs, vals to seq length L
        if xs[0] != 0:
            print(util.get_time(), 'zeroing first map coordinate')
            xs[0] = 0
        if xs[-1] <= L:
            x_edges = np.append(xs, L)
        else:
            vals = vals[xs < L]
            x_edges = np.append(xs[xs < L], L)
        return x_edges, vals

    try:
        u_map = float(u)
        print(util.get_time(), f'using uniform u {u_map}')
    except:
        regions, data = util.read_bedgraph(u)
        edges, u = truncate_map(regions[:, 0], data['u'], L)
        u_map = msprime.RateMap(position=edges, rate=u)
        print(util.get_time(), 'loaded u-map')

    try:
        r_map = float(r)
        print(util.get_time(), f'using uniform r {r_map}')
    except:
        coords, map_vals = util.read_map_file(r)
        map_rates = np.diff(map_vals) / np.diff(coords) / 100
        edges, map_rates = truncate_map(coords[:-1], map_rates, L)
        r_map = msprime.RateMap(position=edges, rate=map_rates)
        print(util.get_time(), 'loaded r-map')

    if isinstance(graph, str):
        graph = demes.load(graph)

    demography = msprime.Demography.from_demes(graph)

    if sampled_demes is None:
        sampled_demes = [d.name for d in graph.demes if d.end_time == 0]

    config = {s: 1 for s in sampled_demes}

    if dtwf_time > 0:
        model = [
            msprime.DiscreteTimeWrightFisher(duration=int(dtwf_time)),
            msprime.StandardCoalescent()
        ]
        print(util.get_time(), f'using DTWF model for first {dtwf_time} g')
    else:
        model = msprime.StandardCoalescent()
        print(util.get_time(), 'using standard Hudson coalescent model')

    print(util.get_time(), 'simulating ancestry')
    ts = msprime.sim_ancestry(
        samples=config,
        ploidy=2,
        demography=demography,
        recombination_rate=r_map,
        discrete_genome=True,
        record_provenance=False,
        sequence_length=L,
        model=model
    )
    print(util.get_time(), 'simulating mutation')
    mts = msprime.sim_mutations(
        ts,
        rate=u_map,
        record_provenance=False
    )

    with open(out_fname, 'w') as file:
        mts.write_vcf(
            file,
            individual_names=sampled_demes,
            contig_id=str(contig_id),
            position_transform=increment1
        )
    print(
        util.get_time(), f'{int(mts.sequence_length)} sites simulated '
        f'on chromosome {contig_id} and saved at {out_fname}'
    )
    return 0


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
    print(util.get_time(), f'mean u in mask: {mean_u}')

    # r setup
    rcoords, rvals = util.read_map_file(args.rmap)
    map_length = rvals[-1]
    map_span = rcoords[-1] - rcoords[0]
    mean_r = map_length / map_span / 100
    map_end_val = map_span * mean_r * 100
    # we need to write a dummy map file for parsing. rate column is not needed
    unif_rmap = f'unif_rmap_{chrom_num}.txt'
    with open(unif_rmap, 'w') as file:
        file.write(
            f'Position(bp)\tRate(cM/Mb)\tMap(cM)\n{int(rcoords[0])}\t0\t0\n'
            f'{int(rcoords[-1])}\t0\t{map_end_val}'
        )

    # simulate!
    sim_fnames = []
    designations = []

    for u_map, u_type in zip([mean_u, args.umap], ['unif-u', 'emp-u']):
        for r_map, r_type in zip([mean_r, args.rmap], ['unif-r', 'emp-r']):

            if (u_type, r_type) not in sim_configs:
                continue

            designation = f'{u_type}-{r_type}'
            designations.append(designation)
            fname = f'{tag}_{designation}.vcf'
            simulate(
                graph,
                fname,
                u=u_map,
                r=r_map,
                L=positions[-1],
                dtwf_time=args.dtwf_time
            )
            sim_fnames.append(fname)

    # then parse
    for sim_fname, designation in zip(sim_fnames, designations):
        for rmap, r_type in zip([unif_rmap, args.rmap], ['unif-r', 'emp-r']):

            if ('emp-u', r_type) in parse_configs:
                dic = parsing.parse_weighted_H2(
                    args.mask,
                    sim_fname,
                    rmap,
                    args.umap,
                    bins=bins,
                    windows=windows,
                    get_denominator=args.count_site_pairs
                )
                np.savez(f'{tag}_{designation}_emp-u-{r_type}.npz', **dic)

            if ('unif-u', r_type) in parse_configs:
                dic = parsing.parse_H2(
                    args.mask,
                    sim_fname,
                    rmap,
                    windows=windows,
                    bins=bins,
                    get_denominator=args.count_site_pairs
                )
                np.savez(f'{tag}_{designation}_unif-u-{r_type}.npz', **dic)


if __name__ == '__main__':
    main()
