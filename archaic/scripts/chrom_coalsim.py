"""
Simulate coalescence on the framework of a real chromosome
"""


import argparse
import demes
import msprime
import numpy as np
from archaic import two_locus


def get_args():
    # get args
    parser = argparse.ArgumentParser()
    parser.add_argument("-g", "--graph_fname", required=True)
    parser.add_argument("-m", "--map_fname", required=True)
    parser.add_argument("-o", "--out_fname", required=True)
    parser.add_argument("-s", "--samples", nargs='*', default=[])
    parser.add_argument("-L", "--L", type=float, default=None)
    parser.add_argument("-c", "--contig", default="0")
    parser.add_argument('-u', "--u", type=float, default=1.35e-8)
    return parser.parse_args()


def main():

    args = get_args()
    if args.L:
        L = args.L
    else:
        map_positions, _ = two_locus.read_map_file(args.map_fname)
        L = map_positions[-1]
    rmap = msprime.RateMap.read_hapmap(args.map_fname, sequence_length=L)
    graph = demes.load(args.yaml_file_name)
    demography = msprime.Demography.from_demes(graph)
    if len(args.samples) > 0:
        samples = args.samples
    else:
        samples = [d for d in graph.demes if d.end_time == 0]
    config = {s: 1 for s in samples}
    #
    ts = msprime.sim_ancestry(
        samples=config,
        ploidy=2,
        demography=demography,
        sequence_length=L,
        recombination_rate=rmap,
        discrete_genome=True
    )
    mts = msprime.sim_mutations(ts, rate=args.u)
    with open(args.out_fname, 'w') as file:
        mts.write_vcf(
            file, individual_names=samples, contig_id=args.contig_id
        )
    return 0


if __name__ == "__main__":
    main()
