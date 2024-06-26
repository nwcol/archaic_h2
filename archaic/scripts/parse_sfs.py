
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


def get_two_sample_SFS(alts):
    # for two samples. i on rows j on cols
    arr = np.zeros((3, 3))
    for i in range(3):
        for j in range(3):
            arr[i, j] = np.count_nonzero(np.all(alts == [i, j], axis=1))
    arr[0, 0] = 0
    return arr


def main():
    # genotype arr has shape L_vcfs, n_samples, 2 
    genotypes = []
    for fname in args.in_fnames:
        _, sample_names, gt = one_locus.read_vcf_file(fname)
        genotypes.append(gt)
    genotypes = np.concatenate(genotypes, axis=0)
    alts = genotypes.sum(2)
    n_samples = len(sample_names)
    kwargs = dict(
        sample_names=sample_names,
        pair_names=utils.get_pair_names(sample_names)
    )
    SFS_arr = []
    for i, j in utils.get_pair_idxs(n_samples):
        SFS_arr.append(get_two_sample_SFS(alts[:, [i, j]]))
    kwargs["sfs_arr"] = np.concatenate(SFS_arr)
    np.savez(args.out_fname, **kwargs)
    return 0


if __name__ == "__main__":
    args = get_args()
    main()
    
