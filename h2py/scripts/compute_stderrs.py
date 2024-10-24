
import argparse
from scipy import stats
import numpy as np

from h2py import inference
from h2py.h2stats_mod import H2stats


def get_args():
    
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-d',
        '--data_fname',
        type=str,
        required=True
    )
    parser.add_argument(
        '-g',
        '--graph_fname',
        type=str,
        required=True
    )
    parser.add_argument(
        '-p',
        '--param_fname',
        type=str,
        required=True
    )
    parser.add_argument(
        '-u',
        type=float,
        required=True
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
    data = H2stats.from_file(args.data_fname, graph=args.graph_fname)
    pnames, p, uncerts = inference.get_uncerts(
        args.graph_fname,
        args.param_fname,
        data,
        bootstraps=None,
        u=None,
        delta=0.01,
        method='GIM'
    )
    conf = 0.95
    z = stats.norm().ppf(0.5 + conf / 2)
    print('parameter\tstderr\t95% confidence')
    for pname, _p, stderr in zip(pnames, p, uncerts):       
        ci = np.format_float_scientific(z * stderr)
        _p = np.format_float_scientific(_p, 2)
        uncert = np.format_float_scientific(uncert, 2)
        print(f'{pname}\t{stderr}\t{_p}-+{ci}')
    return


if __name__ == '__main__':
    main()
    