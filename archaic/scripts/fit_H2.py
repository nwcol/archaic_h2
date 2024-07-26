"""
Fit a demes model to H2 data
"""
import argparse
import demes

from archaic import inference, utils
from archaic.spectra import H2Spectrum


def get_args():
    # get args
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--data_fname', required=True)
    parser.add_argument('-g', '--graph_fname', required=True)
    parser.add_argument('-p', '--options_fname', required=True)
    parser.add_argument('-o', '--out_prefix', required=True)
    parser.add_argument('-u', '--u', type=float, default=1.35e-8)
    parser.add_argument('-H', '--use_H', type=int, default=1)
    parser.add_argument('--max_iter', nargs='*', type=int, default=[1000])
    parser.add_argument('--method', nargs='*', default=['NelderMead'])
    parser.add_argument('-v', '--verbosity', type=int, default=1)
    parser.add_argument('--perturb_graph', type=int, default=0)
    parser.add_argument('--cluster_id', default='')
    parser.add_argument('--process_id', default='')
    return parser.parse_args()


def main():
    #
    args = get_args()
    data = H2Spectrum.from_bootstrap_file(
        args.data_fname, graph=demes.load(args.graph_fname)
    )
    print(utils.get_time(), f'running inference for demes {data.sample_ids}')

    if len(args.method) != len(args.max_iter):
        raise ValueError('')
    tag = inference.get_tag(args.out_prefix, args.cluster_id, args.process_id)
    if args.perturb_graph:
        graph_fname = f'{tag}_init.yaml'
        inference.perturb_graph(
            args.graph_fname, args.options_fname, out_fname=graph_fname
        )
    else:
        graph_fname = args.graph_fname

    for i, method in enumerate(args.method):
        if len(args.method) > 1:
            out_fname = f'{tag}_iter{i + 1}.yaml'
        else:
            out_fname = f'{tag}.yaml'
        inference.fit_H2(
            graph_fname,
            args.options_fname,
            data,
            max_iter=args.max_iter[i],
            method=method,
            u=args.u,
            verbosity=args.verbosity,
            use_H=args.use_H,
            out_fname=out_fname
        )
        graph_fname = out_fname
    return 0


if __name__ == '__main__':
    main()
