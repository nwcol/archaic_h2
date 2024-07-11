"""
Just calls the demesdraw function
"""


import argparse
import demes
import demesdraw
import matplotlib.pyplot as plt
import numpy as np
from archaic import inference


def get_args():
    # get args
    parser = argparse.ArgumentParser()
    parser.add_argument('-g', '--graph_fname', required=True)
    parser.add_argument('-o', '--out_fname', required=True)
    parser.add_argument('-d', '--h2_archive', default=None)
    parser.add_argument('-u', '--u', type=float, default=1.35e-8)
    return parser.parse_args()


def main():
    #
    args = get_args()
    if args.h2_archive:
        ll = inference.compute_graph_log_lik(
            args.graph_fname,
            args.h2_archive,
            u=args.u
        )
        ll_label = f', LL: {np.round(ll, 2)}'
    else:
        ll_label = ''
    title = f'{args.graph_fname} {ll_label}'
    fig, ax = plt.subplots(figsize=(6, 5), layout='constrained')
    demesdraw.tubes(
        demes.load(args.graph_fname),
        title=title,
        inf_ratio=0.1,
        ax=ax
    )
    plt.savefig(args.out_fname, dpi=200)
    return 0


if __name__ == '__main__':
    main()
