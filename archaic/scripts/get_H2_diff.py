
"""
Form (g1 - g2) / g1
"""

import argparse
import demes
import numpy as np
from archaic import inference
from archaic import utils


def get_args():
    # get args
    parser = argparse.ArgumentParser()
    parser.add_argument("-g1", "--graph_fname1", required=True)
    parser.add_argument("-g2", "--graph_fname2", required=True)
    parser.add_argument("-o", "--out_fname", required=True)
    parser.add_argument("-s", "--sample_names", nargs='*', required=True)
    parser.add_argument("-u", "--u", type=float, default=1.35e-8)
    return parser.parse_args()


def main():
    
    r = np.logspace(-6, -2, 50)
    pairs = utils.get_pairs(args.sample_names)
    pair_names = utils.get_pair_names(args.sample_names)
    graph1 = demes.load(args.graph_fname1)
    E_H1, E_H21 = inference.get_H_stats(
        graph1, args.sample_names, pairs, r, args.u, num_method="midpoint"
    )
    graph2 = demes.load(args.graph_fname2)
    E_H2, E_H22 = inference.get_H_stats(
        graph2, args.sample_names, pairs, r, args.u, num_method="midpoint"
    )
    diff_H = (E_H1 - E_H2) / E_H1
    diff_H2 = (E_H21 - E_H22) / E_H21
    np.savez(
        args.out_fname,
        r=r,
        sample_names=args.sample_names,
        pair_names=pair_names,
        H=diff_H,
        H2=diff_H2
    )
    return 0


if __name__ == "__main__":
    args = get_args()
    main()

