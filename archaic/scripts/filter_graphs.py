"""
Copies graphs with fopt values above a threshold or percentile into a directory
"""


import argparse
import demes
import numpy as np


def get_args():
    # get args
    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--threshold", default=None, type=float)
    parser.add_argument('-p', '--percentile', default=None, type=float)
    parser.add_argument("-g", "--graph_fnames", nargs='*')
    parser.add_argument("-o", "--out_path", required=True)
    return parser.parse_args()


def main():
    #
    args = get_args()
    if args.threshold is None and args.percentile is None:
        raise ValueError('you must provide a threshold or a percentile')
    elif args.threshold is not None and args.percentile is not None:
        raise ValueError('you cannot provide both threshold and percentile')
    out_path = args.out_path.rstrip('/')
    graphs = []
    fopts = []
    for fname in args.graph_fnames:
        graph = demes.load(fname)
        graphs.append(graph)
        fopts.append(float(graph.metadata['opt_info']["fopt"]))
    if args.threshold is not None:
        for i, fopt in enumerate(fopts):
            if fopt > args.threshold:
                fname = args.graph_fnames[i]
                print(fname, '\t', fopt)
                basename = fname.split('/')[-1]
                out_fname = f"{out_path}/{basename}"
                demes.dump(graphs[i], out_fname)
    elif args.percentile is not None:
        threshold = np.quantile(fopts, args.percentile, method='linear')
        print(f'threshold: {threshold}')
        indices = np.nonzero(fopts > threshold)[0]
        for i in indices:
            fname = args.graph_fnames[i]
            print(fname, '\t', fopts[i])
            basename = fname.split('/')[-1]
            out_fname = f"{out_path}/{basename}"
            demes.dump(graphs[i], out_fname)
    return 0


if __name__ == "__main__":
    main()
