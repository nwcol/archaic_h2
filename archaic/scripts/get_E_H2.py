
"""
Write expected statistics to file
"""

import argparse
import demes
import numpy as np
from archaic import inference
from archaic import utils


def get_args():
    # get args
    parser = argparse.ArgumentParser()
    parser.add_argument("-g", "--graph_fname", required=True)
    parser.add_argument("-o", "--out_fname", required=True)
    parser.add_argument("-s", "--sample_names", nargs='*', required=True)
    parser.add_argument("-u", "--u", type=float, default=1.35e-8)
    return parser.parse_args()


def main():
    
    r = np.logspace(-6, -2, 40)
    graph = demes.load(args.graph_fname)
    pairs = utils.get_pairs(args.sample_names)
    pair_names = utils.get_pair_names(args.sample_names)
    E_H, E_H2 = inference.get_H_stats(
        graph, args.sample_names, pairs, r, args.u, num_method="midpoint"
    )
    np.savez(
        args.out_fname,
        r=r,
        sample_names=args.sample_names,
        pair_names=pair_names,
        H=E_H,
        H2=E_H2
    )
    return 0


if __name__ == "__main__":
    args = get_args()
    main()

