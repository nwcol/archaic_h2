"""
Each individual is treated as a population
"""


import argparse
import numpy as np
from archaic import one_locus
from archaic import utils


def get_args():
    # get args
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--in_fnames", nargs='*', required=True)
    parser.add_argument("-o", "--out_fname", required=True)
    return parser.parse_args()


def parse(in_fnames, out_fname):
    #
    sample_names = None
    genotypes = []
    for fname in in_fnames:
        _, sample_names, gt = one_locus.read_vcf_file(fname)
        genotypes.append(gt)
    genotypes = np.concatenate(genotypes, axis=0)
    alts = genotypes.sum(2)
    n_samples = len(sample_names)
    sfs_arrs = []
    for i, j in utils.get_pair_idxs(n_samples):
        sfs_arrs.append(one_locus.two_sample_sfs_matrix(alts[:, [i, j]]))
    sfs_arr = np.stack(sfs_arrs)
    kwargs = dict(
        sample_names=sample_names,
        pair_names=utils.get_pair_names(sample_names),
        sfs=sfs_arr
    )
    np.savez(out_fname, **kwargs)


def main():
    #
    args = get_args()
    parse(args.in_fnames, args.out_fname)
    return 0


if __name__ == "__main__":
    main()
