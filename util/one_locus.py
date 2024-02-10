
# Some functions for computing one-locus genetic statistics.

import matplotlib.pyplot as plt
import matplotlib
import numpy as np
from util import sample_sets


if __name__ == "__main__":
    plt.rcParams['figure.dpi'] = 100
    matplotlib.use('Qt5Agg')


def compute_pi(sample_set, *sample_ids, window=None):
    """
    Compute nucleotide diversity. If no *sample_ids are provided, compute the
    statistic for every sample_id in the sample_set

    :return: a dictionary mapping sample_ids to diversities
    """
    if len(sample_ids) == 0:
        sample_ids = sample_set.sample_ids
    if not window:
        window = sample_set.big_window
    L = sample_set.get_window_position_count(window)
    pi_dict = {sample_id: None for sample_id in sample_ids}
    for sample_id in pi_dict:
        pi = sample_set.get_n_sample_het_sites(sample_id, window=window) / L
        pi_dict[sample_id] = pi
    return pi_dict


def compute_site_diff_probs(genotypes_x, genotypes_y):
    """
    At each site, compute the probability that one allele sampled from x
    differs from one allele sampled from y
    """
    probs = np.sum(genotypes_x[:, 0][:, np.newaxis] != genotypes_y, axis=1)\
        + np.sum(genotypes_x[:, 1][:, np.newaxis] != genotypes_y, axis=1)
    probs = probs / 4
    return probs


def compute_pi_xy(sample_set, *sample_ids, window=None):
    """
    Compute nucleotide divergence

    :return: a dictionary mapping sample pairs to divergences
    """
    if len(sample_ids) == 0:
        sample_ids = sample_set.sample_ids
    if not window:
        window = sample_set.big_window
    sample_pairs = enumerate_pairs(sample_ids)
    L = sample_set.get_window_position_count(window)
    win_idx = sample_set.get_window_variant_idx(window)
    pi_xy_dict = {sample_pair: None for sample_pair in sample_pairs}
    for sample_pair in pi_xy_dict:
        sample_id_x, sample_id_y = sample_pair
        probs = compute_site_diff_probs(
            sample_set.genotypes[sample_id_x][win_idx],
            sample_set.genotypes[sample_id_y][win_idx]
        )
        pi_xy = np.sum(probs) / L
        pi_xy_dict[sample_pair] = pi_xy
    return pi_xy_dict


def old_compute_site_diff_probs(genotypes_x, genotypes_y):
    """
    Given two arrays of genotypes: for each site, compute the probability that
    one allele randomly sampled from genotype x will differ from one allele
    randomly sampled from genotype y

    :param genotypes_x: 2d numpy array of genotypes
    :param genotypes_y: 2d numpy array of genotypes
    """
    n_sites = len(genotypes_x)
    if len(genotypes_y) != n_sites:
        raise ValueError("genotype lengths don't match!!!")
    probs = np.zeros(n_sites, dtype=np.float64)
    for row_idx in np.arange(n_sites):
        x_alleles = genotypes_x[row_idx]
        y_alleles = genotypes_y[row_idx]
        probs[row_idx] = (
            np.sum(x_alleles[0] != y_alleles)
            + np.sum(x_alleles[1] != y_alleles)
        ) / 4
    return probs


def compute_approx_pi_xy(sample_set, *sample_ids):
    """
    Compute an approximation to the estimator for nucleotide divergence using
    alternate allele counts (this is inaccurate for triallelic sites)

    :return: a dictionary mapping sample pairs to divergences
    """
    if len(sample_ids) == 0:
        sample_ids = sample_set.sample_ids
    sample_pairs = enumerate_pairs(sample_ids)
    L = sample_set.n_positions
    pi_xy_dict = {sample_pair: None for sample_pair in sample_pairs}
    for sample_pair in pi_xy_dict:
        sample_id_x, sample_id_y = sample_pair
        alts_x = sample_set.get_sample_alt_counts(sample_id_x)
        alts_y = sample_set.get_sample_alt_counts(sample_id_y)
        pi_xy = np.sum((2 - alts_x) * alts_y + (2 - alts_y) * alts_x) / (L * 4)
        pi_xy_dict[sample_pair] = pi_xy
    return pi_xy_dict


def enumerate_pairs(items):
    """
    Return a list of 2-tuples containing every pair of objects in items
    """
    n = len(items)
    pairs = []
    for i in np.arange(n):
        for j in np.arange(i + 1, n):
            pairs.append((items[i], items[j]))
    return pairs
