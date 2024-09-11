"""

"""


import argparse
import demes
import matplotlib.pyplot as plt
import matplotlib
import moments
import numpy as np
from archaic import util


def get_args():
    # get args
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--sfs_archive', required=True)
    parser.add_argument('-o', '--out_fname', required=True)
    return parser.parse_args()


def main():
    #
    args = get_args()
    archive = np.load(args.sfs_archive)
    samples = archive['samples']
    sfs = moments.Spectrum(archive['SFS'], pop_ids=samples)
    z_lim = sfs.max()
    n = len(samples)
    n_axs = util.n_choose_2(n)
    cols = 5
    rows = int(np.ceil(n_axs / cols))
    fig, axs = plt.subplots(
        rows, cols, figsize=(2 * cols, 2 * rows), layout='constrained'
    )
    axs = axs.flat
    idxs = utils.get_pair_idxs(n)
    for k in range(len(idxs)):
        ax = axs[k]
        i, j = idxs[k]
        sample_i = samples[i]
        sample_j = samples[j]
        mask = [k for k in range(10) if k != i and k != j]
        marginal_sfs = sfs.marginalize(mask)
        ax.pcolormesh(
            marginal_sfs,
            cmap=matplotlib.cm.gnuplot,
            vmin=0,
            vmax=z_lim
        )
        ax.set_title(f'{sample_i},{sample_j}')
    plt.savefig(args.out_fname, dpi=200)
    return 0


if __name__ == '__main__':
    main()
