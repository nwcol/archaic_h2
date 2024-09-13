"""
Simulate coalescence on a chromosome defined by recombination and mutation
rate-maps
"""
import argparse
import demes
import msprime

from archaic import util, simulation


def get_args():
    # get args
    parser = argparse.ArgumentParser()
    parser.add_argument("-g", "--graph_fname", required=True)
    parser.add_argument("-o", "--out_fname", required=True)
    parser.add_argument("-s", "--sampled_demes", nargs='*', default=None)
    parser.add_argument("-u", "--u_fname", required=True)
    parser.add_argument("-r", "--r_fname", required=True)
    parser.add_argument("-c", "--contig_id", default="0")
    parser.add_argument('-L', '--seq_length', type=int, default=None)
    return parser.parse_args()


def main():
    #
    args = get_args()
    simulation.simulate_chromosome(
        args.graph_fname,
        args.out_fname,
        u=args.u_fname,
        r=args.r_fname,
        sampled_demes=args.sampled_demes,
        contig_id=args.contig_id,
        L=args.seq_length
    )
    print(
        util.get_time(),
        f'simulation completed and saved at {args.out_fname}'
    )
    return 0


if __name__ == "__main__":
    main()
