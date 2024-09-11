"""
SFS + H2
"""
import argparse
import demes
import moments
import numpy as np

from archaic import inference, util
from archaic.spectra import H2Spectrum


def get_args():
    # get args
    parser = argparse.ArgumentParser()
    parser.add_argument('-H2', '--H2_fname', required=True)
    parser.add_argument('-SFS', '--SFS_fname', required=True)
    parser.add_argument('-g', '--graph_fname', required=True)
    parser.add_argument('-p', '--params_fname', required=True)
    parser.add_argument('-o', '--out_prefix', required=True)
    parser.add_argument('-u', '--u', type=float, default=None)
    parser.add_argument('--max_iter', nargs='*', type=int, default=[200])
    parser.add_argument('--method', nargs='*', default=['Powell'])
    parser.add_argument('-v', '--verbosity', type=int, default=1)
    parser.add_argument('--perturb_graph', type=int, default=0)
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
    if len(args.method) != len(args.max_iter):
        raise ValueError('')
    tag = get_tag(args.out_prefix, args.cluster_id, args.process_id)
    if args.perturb_graph:
        graph_fname = f'{tag}_init.yaml'
        inference.perturb_graph(
            args.graph_fname, args.params_fname, out_fname=graph_fname
        )
    else:
        graph_fname = args.graph_fname

    # read H2 data
    H2_data = H2Spectrum.from_bootstrap_file(
        args.H2_fname, graph=demes.load(args.graph_fname)
    )
    pop_ids = H2_data.sample_ids
    # read SFS data
    SFS_data, L = inference.read_SFS(args.SFS_fname, pop_ids)

    print(
        util.get_time(),
        f'running inference for demes {H2_data.sample_ids}'
    )
    for i, method in enumerate(args.method):
        if len(args.method) > 1:
            out_fname = f'{tag}_iter{i + 1}.yaml'
        else:
            out_fname = f'{tag}.yaml'
        inference.fit_composite(
            graph_fname,
            args.params_fname,
            H2_data,
            SFS_data,
            L=L,
            u=args.u,
            max_iter=args.max_iter[i],
            verbosity=args.verbosity,
            method=method,
            out_fname=out_fname
        )
    return 0


if __name__ == '__main__':
    main()
