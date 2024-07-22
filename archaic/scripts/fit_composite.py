"""
SFS + H2
"""

import argparse
import demes
import moments
import numpy as np
from archaic import inference, utils
from archaic.spectra import H2Spectrum


def get_args():
    # get args
    parser = argparse.ArgumentParser()
    parser.add_argument('-H2', '--H2_fname', required=True)
    parser.add_argument('-SFS', '--SFS_fname', required=True)
    parser.add_argument('-g', '--graph_fname', required=True)
    parser.add_argument('-p', '--params_fname', required=True)
    parser.add_argument('-o', '--out_prefix', required=True)
    parser.add_argument('-u', '--u', type=float, default=1.35e-8)
    parser.add_argument('-max', '--max_iter', nargs='*', type=int, default=[200])
    parser.add_argument('-opt', '--opt_methods', nargs='*', default=['NelderMead'])
    parser.add_argument('-v', '--verbosity', type=int, default=1)
    parser.add_argument('--permute_graph', type=int, default=0)
    parser.add_argument('--cluster_id', default='')
    parser.add_argument('--process_id', default='')
    return parser.parse_args()


def get_tag(prefix, cluster, process):
    # get a string naming an output .yaml file
    c = p = ''
    if len(cluster) > 0:
        c = f'_{cluster}'
    if len(process) > 0:
        p = f'_{process}'
    tag = f'{prefix}{c}{p}'
    return tag


def main():
    #
    args = get_args()
    if len(args.opt_methods) != len(args.max_iter):
        raise ValueError('')
    tag = get_tag(args.out_prefix, args.cluster_id, args.process_id)
    if args.permute_graph:
        graph_fname = f'{tag}_init.yaml'
        inference.perturb_graph(
            args.graph_fname, args.params_fname, graph_fname
        )
    else:
        graph_fname = args.graph_fname
    # read SFS data
    SFS_file = np.load(args.SFS_fname)
    pop_ids = list(SFS_file['samples'])
    data = moments.Spectrum(SFS_file['SFS'], pop_ids=pop_ids)
    deme_names = [d.name for d in demes.load(args.graph_fname).demes]
    marg_idx = []
    for i, pop_id in enumerate(pop_ids):
        if pop_id in deme_names:
            pass
        else:
            marg_idx.append(i)
    SFS_data = data.marginalize(marg_idx)
    L = SFS_file['n_sites']
    # read H2 data
    H2_data = H2Spectrum.from_bootstrap_file(
        args.H2_fname, graph=demes.load(args.graph_fname)
    )
    print(utils.get_time(), f'running inference for demes {H2_data.sample_ids}')
    for i, opt_method in enumerate(args.opt_methods):
        graph, opt_info = inference.optimize_super_composite(
            graph_fname,
            args.params_fname,
            H2_data,
            SFS_data,
            L,
            args.max_iter[i],
            verbosity=args.verbosity,
            u=args.u,
            opt_method=opt_method
        )
        graph.metadata['opt_info'] = opt_info
        out_fname = f'{tag}_iter{i + 1}.yaml'
        demes.dump(graph, out_fname)
        graph_fname = out_fname
    return 0


if __name__ == '__main__':
    main()
