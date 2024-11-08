"""
Plot an arbitrary number of H, H2 expectations alongside 0 or 1 empirical vals.
"""
import argparse
import demes
import matplotlib.pyplot as plt
import numpy as np
import pickle
import os

from h2py import h2_parsing, inference, plotting


def get_args():
    # get args
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-d', '--data_files', 
        nargs='*',
        default=[],
        type=str
    )
    parser.add_argument(
        '-g', '--graph_files', 
        nargs='*',
        default=[],
        type=str
    )
    parser.add_argument(
        '--pop_ids',
        nargs='*',
        default=None,
        type=str,
    )
    parser.add_argument(
        '--ylim_0', type=int, default=0
    )
    parser.add_argument(
        '--ratios', type=int, default=0
    )
    parser.add_argument(
        '-H', '--plot_H', type=int, default=1
    )
    parser.add_argument(
        '-o', '--out_file', required=True
    )
    parser.add_argument(
        '-u', '--u', type=float, default=None
    )
    return parser.parse_args()


def main():
    """
    """
    args = get_args()

    if len(args.graph_files) > 0:
        if args.u is not None:
            u = args.u
        else:
            u = None
            for file in args.graph_files:
                g = demes.load(file)
                if 'opt_info' in g.metadata:
                    u = g.metadata['opt_info']['u']
                elif 'u' in g.metadata:
                    u = g.metadata['u']
                break
            if u is None:
                raise ValueError('please provide a u parameter')
    
    # load statistics
    g = args.graph_files[0] if len(args.graph_files) > 0 else None
    datas = []
    labels = []

    for file in args.data_files:
        with open(file, 'rb') as fin:
            dic = pickle.load(fin)
        for key in dic:
            data = h2_parsing.subset_H2(dic[key], graph=g)
            label = os.path.basename(file) + '-' + key
            datas.append(data)
            labels.append(label)

    # load graphs
    pop_ids = args.pop_ids

    if len(datas) > 0:
        data = datas[0]
        models = [inference.moments_H2(g, u=u, data=data) 
                  for g in args.graph_files]
        lls = [np.round(inference.compute_ll(m, data, include_H=False), 2)
               for m in models] 
        labels += [os.path.basename(args.graph_files[i]) + f', ll={lls[i]}'
                   for i in range(len(args.graph_files))]

    else:
        models = [inference.moments_H2(g, u, sampled_demes=pop_ids) 
                  for g in args.graph_files]
        labels += [os.path.basename(g) for g in args.graph_files]
        
    plotting.plot_H2s(
        models=models, 
        datas=datas,
        labels=labels,
        conf=0.95,
        plot_H=True,
        ylim_0=args.ylim_0
    )

    plt.savefig(args.out_file, dpi=244, bbox_inches='tight')
    return 


if __name__ == "__main__":    
    main()
