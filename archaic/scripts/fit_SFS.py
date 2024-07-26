"""

"""
import argparse
import demes
import moments
import numpy as np

from archaic import inference as inference


def get_args():
    # get args
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--data_fname', required=True)
    parser.add_argument('-g', '--graph_fname', required=True)
    parser.add_argument('-p', '--options_fname', required=True)
    parser.add_argument('-o', '--out_prefix', required=True)
    parser.add_argument('-u', '--u', type=float, default=1.35e-8)
    parser.add_argument('--max_iter', nargs='*', type=int, default=[1000])
    parser.add_argument('--method', nargs='*', default=['powell'])
    parser.add_argument('-v', '--verbosity', type=int, default=1)
    parser.add_argument('--perturb_graph', type=int, default=0)
    parser.add_argument('--cluster_id', default='')
    parser.add_argument('--process_id', default='')
    return parser.parse_args()


def main():
    #
    args = get_args()

    ### replace
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
    ###

    tag = inference.get_tag(args.out_prefix, args.cluster_id, args.process_id)
    if args.perturb_graph:
        graph_fname = f'{tag}_init.yaml'
        inference.perturb_graph(
            args.graph_fname, args.params_fname, out_fname=graph_fname
        )
    else:
        graph_fname = args.graph_fname

    for i, method in enumerate(args.method):
        if len(args.method) > 1:
            out_fname = f'{tag}_iter{i + 1}.yaml'
        else:
            out_fname = f'{tag}.yaml'
        inference.fit_SFS(
            graph_fname,
            args.options_fname,
            data,
            uL,
            max_iter=args.max_iter[i],
            method=method,
            verbosity=args.verbosity,
            out_fname=out_fname
        )
    return 0


if __name__ == '__main__':
    main()