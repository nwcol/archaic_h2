"""
Perturb a graph, then run SFS and H2 inference on that graph
"""


import argparse
import demes
import moments
import msprime
import numpy as np
from archaic import inference
from archaic import masks
from archaic import two_locus
from archaic.parsing import parse_H2, bootstrap_H2, parse_SFS


temp_dir = 'temp'


def get_args():
    # get args
    parser = argparse.ArgumentParser()
    parser.add_argument('-H2', '--H2_data_fname', required=True)
    parser.add_argument('-SFS', '--SFS_data_fname', required=True)
    parser.add_argument('-g', '--graph_fname', required=True)
    parser.add_argument('-p', '--params_fname', required=True)
    parser.add_argument('-o', '--out_prefix', required=True)
    parser.add_argument('-max', '--max_iter', nargs='*', type=int, default=[1000])
    parser.add_argument('-opt', '--opt_methods', nargs='*', default=['Powell'])
    parser.add_argument('-v', '--verbosity', type=int, default=1)
    parser.add_argument('-s', '--samples', nargs='*', default=None)
    parser.add_argument('-u', '--u', type=float, default=1.35e-8)
    parser.add_argument('--cluster_id', default='0')
    parser.add_argument('--process_id', default='0')
    return parser.parse_args()


def get_graph_fname(prefix, statistic, cluster, process, i):
    # get a string naming an output .yaml file
    c = p = ''
    if len(cluster) > 0:
        c = f'_{cluster}'
    if len(process) > 0:
        p = f'_{process}'
    fname = f'{prefix}_{statistic}{c}{p}_iter{i}.yaml'
    return fname


def main():
    #
    args = get_args()
    graph = demes.load(args.graph_fname)
    _samples = np.load(args.SFS_data_fname)['samples']
    if args.samples:
        samples = args.samples
    else:
        samples = [d.name for d in graph.demes if d.name in _samples]
    print(f'running inference with samples {samples}')
    graph_fname_0 = get_graph_fname(
        args.out_prefix,
        args.cluster_id,
        args.process_id,
        'permuted',
        0
    )
    inference.perturb_graph(
        args.graph_fname, args.params_fname, graph_fname_0
    )
    # inference with H2
    graph_fname = graph_fname_0
    r_bins, data = inference.read_data(args.H2_data_fname, samples)
    for i, opt_method in enumerate(args.opt_methods):
        H2_graph, opt_info = inference.optimize(
            graph_fname,
            args.params_fname,
            data,
            r_bins,
            args.max_iter[i],
            verbosity=args.verbosity,
            u=args.u,
            use_H=True,
            use_H2=True,
            opt_method=opt_method
        )
        out_fname = get_graph_fname(
            args.out_prefix,
            args.cluster_id,
            args.process_id,
            'H2',
            i + 1
        )
        H2_graph.metadata['opt_info'] = opt_info
        demes.dump(H2_graph, out_fname)
        graph_fname = out_fname
    # inference with SFS
    # opt methods are named a bit differently in moments.Demes.Inference
    graph_fname = graph_fname_0
    opt_methods = {
        'NelderMead': 'fmin',
        'BFGS': None,
        'LBFGSB': 'lbfgsb',
        'Powell': 'powell'
    }
    SFS_file = np.load(args.SFS_data_fname)
    data = moments.Spectrum(SFS_file['SFS'], pop_ids=list(SFS_file['samples']))
    # marginalize over any samples which aren't represented in the graph
    marg_idx = []
    for i, pop_id in enumerate(_samples):
        if pop_id in samples:
            pass
        else:
            marg_idx.append(i)
    if len(marg_idx) > 0:
        data = data.marginalize(marg_idx)
    uL = SFS_file['n_sites'] * args.u
    for i, opt_method in enumerate(args.opt_methods):
        out_fname = get_graph_fname(
            args.out_prefix,
            args.cluster_id,
            args.process_id,
            'SFS',
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
            method=opt_methods[opt_method],
            overwrite=True
        )
        opt_info = dict(
            method=opt_methods[opt_method],
            fopt=-LL,
            iters=None,
            func_calls=None,
            warnflag=None
        )
        graph = demes.load(out_fname)
        graph.metadata['opt_info'] = opt_info
        demes.dump(graph, out_fname)
        graph_fname = out_fname
    return 0


if __name__ == '__main__':
    main()
