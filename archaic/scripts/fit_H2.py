"""
Fit a demes model to H2 data
"""
import argparse
import demes

from archaic import inference, util
from archaic.spectra import H2Spectrum


def get_args():
    # get args
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--data_fname', required=True)
    parser.add_argument('-g', '--graph_fname', required=True)
    parser.add_argument('-p', '--options_fname', required=True)
    parser.add_argument('-o', '--out_fname', required=True)
    parser.add_argument('-u', '--u', type=float, default=None)
    parser.add_argument('-H', '--use_H', type=int, default=0)
    parser.add_argument('--max_iters', nargs='*', type=int, default=[1000])
    parser.add_argument('--method', nargs='*', default='Powell')
    parser.add_argument('-v', '--verbosity', type=int, default=10)
    parser.add_argument('--perturb_graph', type=int, default=0)
    return parser.parse_args()


def main():
    #
    args = get_args()
    data = H2Spectrum.from_bootstrap_file(
        args.data_fname, graph=demes.load(args.graph_fname)
    )
    if args.perturb_graph:
        graph_fname = f'{args.out_fname.replace(".yaml", "")}_init.yaml'
        inference.perturb_graph(
            args.graph_fname, args.options_fname, out_fname=graph_fname
        )
    else:
        graph_fname = args.graph_fname
    out_fname = f'{args.out_fname}.yaml'
    inference.fit_H2(
        graph_fname,
        args.options_fname,
        data,
        max_iter=args.max_iters,
        method=args.method,
        u=args.u,
        verbosity=args.verbosity,
        use_H=args.use_H,
        out_fname=out_fname
    )
    return 0


if __name__ == '__main__':
    main()
