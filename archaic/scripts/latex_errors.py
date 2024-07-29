"""
just an experiment rn. prints to stdout
"""
import argparse
import demes
import matplotlib
import matplotlib.pyplot as plt
import moments
import numpy as np

from archaic import inference
from archaic.spectra import H2Spectrum


def get_args():
    # get args
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--params_fname', required=True)
    parser.add_argument('-g', '--graph_fname', required=True)
    parser.add_argument('-d', '--data_fname', required=True)
    parser.add_argument('-o', '--out_fname', required=True)
    return parser.parse_args()


def get_errs(graph_fname, options_fname, data_fname, method='GIM', u=1.35e-8):
    #
    delta = 0.1
    graph = demes.load(graph_fname)
    data = H2Spectrum.from_bootstrap_file(data_fname, graph=graph)
    if method == 'GIM':
        file = np.load(data_fname)
        n = len(file['H2_dist'])
        bootstraps = [
            H2Spectrum.from_bootstrap_distribution(
                data_fname, i, sample_ids=data.sample_ids
            ) for i in range(n)
        ]
    else:
        bootstraps = None
    pnames, p0, std_errs = inference.get_uncerts(
        graph_fname,
        options_fname,
        data,
        bootstraps=bootstraps,
        u=u,
        delta=delta,
        method=method
    )
    return std_errs


def main():
    #
    args = get_args()
    builder = moments.Demes.Inference._get_demes_dict(args.graph_fname)
    options = moments.Demes.Inference._get_params_dict(args.params_fname)
    par_names, params0, _, __ = \
        moments.Demes.Inference._set_up_params_and_bounds(options, builder)
    std_errs = get_errs(
        args.graph_fname, args.params_fname, args.data_fname, method='GIM'
    )
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
