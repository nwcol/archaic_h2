
import argparse
import demes
import msprime
import numpy as np


def get_args():
    # get args
    parser = argparse.ArgumentParser()
    parser.add_argument("-g", "--graph_fname", required=True)
    parser.add_argument("-o", "--out_fname", required=True)
    parser.add_argument("-s", "--samples", nargs='*', required=True)
    parser.add_argument("-L", type=float, required=True)
    parser.add_argument("-u", "--u", type=float, default=1.35e-8)
    parser.add_argument("-r", "--r", type=float, default=1e-8)
    parser.add_argument("-c", "--contig", default="0")
    return parser.parse_args()


def main():
    #
    demog = msprime.Demography.from_demes(demes.load(args.graph_fname))
    config = {s: 1 for s in args.samples}
    ts = msprime.sim_ancestry(
        samples=config,
        ploidy=2,
        demography=demog,
        sequence_length=int(args.L),
        recombination_rate=args.r,
        discrete_genome=True
    )
    mts = msprime.sim_mutations(ts, rate=args.u)
    with open(args.out_fname, 'w') as file:
        mts.write_vcf(
            file, individual_names=args.samples, contig_id=args.contig
        )
    return 0


if __name__ == "__main__":
    args = get_args()
    main()

