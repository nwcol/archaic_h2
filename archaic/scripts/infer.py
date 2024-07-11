"""

"""

import argparse
import demes
from archaic import inference
from archaic import utils


def get_args():
    # get args
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--boot_fname', required=True)  
    parser.add_argument('-g', '--graph_fname', required=True)
    parser.add_argument('-p', '--params_fname', required=True)
    parser.add_argument('-o', '--out_prefix', required=True)
    parser.add_argument('-max', '--max_iters', nargs='*', type=int, default=[1000])
    parser.add_argument('-v', '--verbosity', type=int, default=1)
    parser.add_argument('-opt', '--opt_methods', nargs='*', default=['NelderMead'])
    parser.add_argument('-u', '--u', type=float, default=1.35e-8)
    parser.add_argument('-H', '--use_H', type=int, default=1)
    parser.add_argument('-H2', '--use_H2', type=int, default=1)
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
    sample_names = inference.scan_names(args.graph_fname, args.boot_fname)
    print(utils.get_time(), f'running inference for demes {sample_names}')
    if len(args.opt_methods) != len(args.max_iters):
        raise ValueError('')
    r_bins, data = inference.read_data(args.boot_fname, sample_names)
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
        graph, opt_info = inference.optimize(
            graph_fname,
            args.params_fname,
            data,
            r_bins,
            args.max_iters[i],
            verbosity=args.verbosity,
            u=args.u,
            use_H=args.use_H,
            use_H2=args.use_H2,
            opt_method=opt_method
        )
        graph.metadata['opt_info'] = opt_info
        out_fname = get_graph_fname(
            args.out_prefix,
            args.cluster_id,
            args.process_id,
            i + 1
        )
        demes.dump(graph, out_fname)
        graph_fname = out_fname
    return 0


if __name__ == '__main__':
    main()
