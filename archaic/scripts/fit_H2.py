"""

"""

import argparse
import demes
from archaic import inference
from archaic import utils
from archaic.spectra import H2Spectrum


def get_args():
    # get args
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--data_fname', required=True)
    parser.add_argument('-g', '--graph_fname', required=True)
    parser.add_argument('-p', '--params_fname', required=True)
    parser.add_argument('-o', '--out_prefix', required=True)
    parser.add_argument('-max', '--max_iter', nargs='*', type=int, default=[1000])
    parser.add_argument('-v', '--verbosity', type=int, default=1)
    parser.add_argument('-opt', '--opt_methods', nargs='*', default=['NelderMead'])
    parser.add_argument('-u', '--u', type=float, default=1.35e-8)
    parser.add_argument('-H', '--use_H', type=int, default=1)
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
    data = H2Spectrum.from_bootstrap_file(
        args.data_fname, graph=demes.load(args.graph_fname)
    )
    print(utils.get_time(), f'running inference for demes {data.sample_ids}')
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
    for i, opt_method in enumerate(args.opt_methods):
        graph, opt_info = inference.optimize_H2(
            graph_fname,
            args.params_fname,
            data,
            max_iter=args.max_iter[i],
            opt_method=opt_method,
            u=args.u,
            verbosity=args.verbosity,
            use_H=args.use_H
        )
        graph.metadata['opt_info'] = opt_info
        out_fname = f'{tag}_iter{i + 1}.yaml'
        demes.dump(graph, out_fname)
        graph_fname = out_fname
    return 0


if __name__ == '__main__':
    main()
