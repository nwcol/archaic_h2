"""

"""


import argparse
import demes
import moments
import numpy as np
from archaic import inference


def get_args():
    # get args
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--data_fname', required=True)
    parser.add_argument('-g', '--graph_fname', required=True)
    parser.add_argument('-p', '--params_fname', required=True)
    parser.add_argument('-o', '--out_prefix', required=True)
    parser.add_argument('-max', '--max_iter', nargs='*', type=int, default=[1000])
    parser.add_argument('-opt', '--opt_methods', nargs='*', default=['fmin'])
    parser.add_argument('-v', '--verbosity', type=int, default=1)
    parser.add_argument('-u', '--u', type=float, default=1.35e-8)
    parser.add_argument('--permute_graph', type=int, default=0)
    parser.add_argument('--cluster_id', default='')
    parser.add_argument('--process_id', default='')
    return parser.parse_args()


def get_graph_fname(prefix, cluster, process, i):
    # get a string naming an output .yaml file
    c = p = ''
    if len(cluster) > 0:
        c = f'_{cluster}'
    if len(process) > 0:
        p = f'_{process}'
    fname = f'{prefix}{c}{p}_iter{i}.yaml'
    return fname


def main():
    #
    args = get_args()
    SFS_file = np.load(args.data_fname)
    pop_ids = list(SFS_file['samples'])
    data = moments.Spectrum(SFS_file['SFS'], pop_ids=pop_ids)
    deme_names = [d.name for d in demes.load(args.graph_fname).demes]
    marg_idx = []
    for i, pop_id in enumerate(pop_ids):
        if pop_id in deme_names:
            pass
        else:
            marg_idx.append(i)
    data = data.marginalize(marg_idx)
    uL = SFS_file['n_sites'] * args.u
    if args.permute_graph:
        graph_fname = get_graph_fname(
            args.out_prefix,
            args.cluster_id,
            args.process_id,
            0
        )
        inference.permute_graph(
            args.graph_fname, args.params_fname, graph_fname
        )
    else:
        graph_fname = args.graph_fname
    for i, opt_method in enumerate(args.opt_methods):
        out_fname = get_graph_fname(
            args.out_prefix,
            args.cluster_id,
            args.process_id,
            i + 1
        )
        _, __, LL = moments.Demes.Inference.optimize(
            graph_fname,
            args.params_fname,
            data,
            maxiter=args.max_iter[i],
            verbose=args.verbosity,
            uL=uL,
            log=False,
            output=out_fname,
            method=opt_method,
            overwrite=True
        )
        opt_info = dict(
            method=opt_method,
            fopt=-LL,
            iters=None,
            func_calls=None,
            warnflag=None,
            inferred_with='SFS'
        )
        graph = demes.load(out_fname)
        graph.metadata['opt_info'] = opt_info
        demes.dump(graph, out_fname)
        graph_fname = out_fname
    return 0


if __name__ == '__main__':
    main()