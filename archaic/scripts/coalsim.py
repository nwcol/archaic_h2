"""
Simulate coalescence on a chromosome with uniform recombination
"""
import argparse
import demes
import msprime

from archaic import utils, simulation


def get_args():
    # get args
    parser = argparse.ArgumentParser()
    parser.add_argument("-g", "--graph_fname", required=True)
    parser.add_argument("-o", "--out_fname", required=True)
    parser.add_argument("-L", type=str)
    parser.add_argument("-s", "--sampled_demes", nargs='*', default=None)
    parser.add_argument("-u", "--u", type=str, default='1.35e-8')
    parser.add_argument("-r", "--r", type=str, default='1e-8')
    parser.add_argument("-c", "--contig_id", default="0")
    return parser.parse_args()


def main():
    #
    args = get_args()
    simulation.simulate(
        args.graph_fname,
        L=args.L,
        u=args.u,
        r=args.r,
        sampled_demes=args.sampled_demes,
        contig_id=args.contig_id,
        out_fname=args.out_fname
    )
    print(
        utils.get_time(),
        f'simulation completed and saved at {args.out_fname}'
    )
    return 0


if __name__ == "__main__":
    main()
