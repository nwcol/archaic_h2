"""
Make a scatter plot of parameters from several demes graphs.
"""
import argparse
import demes
import matplotlib.pyplot as plt
from moments.Demes import Inference
import numpy as np

from h2py import plotting


def get_args():
    # get args
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-g', '--graph_files', 
        nargs='*',
        default=[],
        type=str,
        required=True
    )
    parser.add_argument(
        '-p', '--param_file', 
        type=str,
        required=True
    )
    parser.add_argument(
        '-o', '--out_file',
        type=str,
        required=True
    )
    parser.add_argument(
        '--params',
        nargs='*',
        default=None
    )
    parser.add_argument(
        '--min_ll',
        type=float,
        default=None
    )
    return parser.parse_args()


def main():
    """
    
    """
    args = get_args()

    # load 
    options = Inference._get_params_dict(args.param_file)
    ps = []
    lls = []

    for file in args.graph_files:
        graph  = demes.load(file)
        if 'fopt' not in graph.metadata['opt_info']:
            raise ValueError('graphs need `opt_info` metadata')
        ll = graph.metadata['opt_info']['fopt']
        if args.min_ll:
            if ll < args.min_ll:
                continue
        lls.append(ll)

        builder = Inference._get_demes_dict(file)
        names, p, *_ = Inference._set_up_params_and_bounds(options, builder)

        if args.params is not None:
            idx = []
            for param in args.params:
                if param not in names:
                    raise ValueError(f'parameter {param} not in file')
                idx.append(names.index(param))
            names = [names[i] for i in idx]
            p = [p[i] for i in idx]

        ps.append(p)
    
    p_arr = np.array(ps)
    lls = np.array(lls)

    plotting.plot_params(
        names,
        p_arr,
        lls,
    )
    plt.savefig(args.out_file, dpi=244, bbox_inches='tight')

    return


if __name__ == '__main__':
    main()
