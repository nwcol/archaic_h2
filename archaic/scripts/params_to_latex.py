"""
just an experiment rn. prints to stdout
"""


import argparse
import matplotlib
import matplotlib.pyplot as plt
import moments
import numpy as np


def get_args():
    # get args
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--params_fname', required=True)
    parser.add_argument('-g', '--graph_fname', required=True)
    parser.add_argument('-o', '--out_fname', required=True)
    return parser.parse_args()


def main():
    #
    args = get_args()
    builder = moments.Demes.Inference._get_demes_dict(args.graph_fname)
    options = moments.Demes.Inference._get_params_dict(args.params_fname)
    par_names, params0, _, __ = \
        moments.Demes.Inference._set_up_params_and_bounds(options, builder)
    expression = (
        r'\begin{tabular}{ |r|r| } \hline '
        r'Parameter & Value \\'
        r'\hline '
    )
    for i, par_name in enumerate(par_names):
        if params0[i] < 1:
            coeff, exponent = np.format_float_scientific(params0[i], precision=2).split('e')
            exponent = '{' + exponent.replace('0', '') + '}'
            val = rf'{coeff} \cdot 10^{exponent}'
        else:
            val = np.round(params0[i], 0).astype(int)
        if '_' in par_name:
            idx = par_name.index('_')
            par_name = par_name[:idx] + '_{' + f'{par_name[idx + 1:]}' + '}'
        expression += rf'${par_name}$ & ${val}$ \\'
    expression += r'\hline \end{tabular}'
    print(expression)
    plt.rcParams['text.usetex'] = True
    matplotlib.rcParams["mathtext.fontset"] = "cm"
    matplotlib.rcParams["font.family"] = "cm"
    fig = plt.figure()
    fig.text(
        x=0.5,
        y=0.5,
        s=expression,
        horizontalalignment="center",
        verticalalignment="center",
        fontsize=16,
    )
    plt.savefig(args.out_fname, dpi=200, bbox_inches='tight')


if __name__ == '__main__':
    main()
