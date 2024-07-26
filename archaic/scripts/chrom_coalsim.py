"""
Simulate coalescence on the framework of a real chromosome
"""
import argparse

from archaic import simulation


def get_args():
    # get args
    parser = argparse.ArgumentParser()
    parser.add_argument("-g", "--graph_fname", required=True)
    parser.add_argument("-r", "--map_fname", required=True)
    parser.add_argument("-o", "--out_fname", required=True)
    parser.add_argument("-s", "--sample_ids", nargs='*', default=None)
    parser.add_argument("-c", "--contig", default="0")
    parser.add_argument('-u', "--u", type=float, default=1.35e-8)
    return parser.parse_args()


def main():
    #
    args = get_args()
    simulation.simulate_chrom(
        args.graph_fname,
        args.map_fname,
        u=args.u,
        sample_ids=args.sample_ids,
        out_fname=args.out_fname,
        contig=args.contig
    )
    return 0


if __name__ == "__main__":
    main()
