
import argparse
import demes
from scipy import stats
import numpy as np
import pickle

from h2py import inference, h2_parsing


def get_args():
    
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-d',
        '--data_files',
        nargs='*',
        type=str,
        required=True
    )
    parser.add_argument(
        '-g',
        '--graph_file',
        type=str,
        required=True
    )
    parser.add_argument(
        '-p',
        '--param_file',
        type=str,
        required=True
    )
    parser.add_argument(
        '-u',
        type=float,
        default=None
    )
    # optional args
    parser.add_argument(
        '--include_H',
        type=int,
        default=0
    )
    return parser.parse_args()


def main():
    """
    
    """
    args = get_args()

    if args.u:
        u = args.u
    else:
        g = demes.load(args.graph_file)
        if 'opt_info' in g.metadata:
            u = g.metadata['opt_info']['u']
        elif 'u' in g.metadata:
            u = g.metadata['u']
        else:
            raise ValueError('please provide a `u` parameter')

    graph = demes.load(args.graph_file)
    regions = {}

    for in_file in args.data_files:
        with open(in_file, 'rb') as fin:
            file_regions = pickle.load(fin)
        for label in file_regions:
            if label in regions:
                raise ValueError(f'{label} occurs twice in input')
            else:
                subset = h2_parsing.subset_H2(file_regions[label], graph=graph)
                regions[label] = subset

    data, raw_reps = h2_parsing.get_bootstrap_reps(regions, mut_weighted=True)

    reps = []
    for raw in raw_reps:
        rep = {
            'pop_ids': data['pop_ids'],
            'bins': data['bins'],
            'means': raw,
            'covs': data['covs']
        }
        reps.append(rep)

    pnames, p, uncerts = inference.compute_uncerts(
        args.graph_file,
        args.param_file,
        data,
        bootstrap_reps=reps,
        u=u,
        delta=0.01,
        method='GIM'
    )

    conf = 0.95
    z = stats.norm().ppf(0.5 + conf / 2)
    print('parameter\tstderr\t95% confidence')

    for pname, _p, stderr in zip(pnames, p, uncerts):       
        ci = np.format_float_scientific(z * stderr, 2)
        _p = np.format_float_scientific(_p, 2)
        print(f'{pname}\t{stderr}\t{_p} + {ci}')
    return


if __name__ == '__main__':
    main()
    