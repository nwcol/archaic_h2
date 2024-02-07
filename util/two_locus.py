
# Functions for computing two-locus genetic statistics

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import time
from util import sample_sets
from util import map_util


if __name__ == "__main__":
    plt.rcParams['figure.dpi'] = 100
    matplotlib.use('Qt5Agg')


r_bin_edges = np.array([0, 1e-7, 2e-7, 5e-7, 1e-6, 2e-6, 5e-6, 1e-5, 2e-5,
                        5e-5, 1e-4, 2e-4, 5e-4, 1e-3, 2e-3, 5e-3, 1e-2, 2e-2,
                        5e-2, 1e-1], dtype=np.float64)

r_bin_mids = np.array([5.0e-8, 1.5e-7, 3.5e-7, 7.5e-7, 1.5e-6, 3.5e-6, 7.5e-6,
                       1.5e-5, 3.5e-5, 7.5e-5, 1.5e-4, 3.5e-4, 7.5e-4, 1.5e-3,
                       3.5e-3, 7.5e-3, 1.5e-2, 3.5e-2, 7.5e-2],
                      dtype=np.float64)

r_bin_right_edges = r_bin_edges[1:]


ex = sample_sets.UnphasedSampleSet.get_chr(22)


def get_max_right_idx(map_values, max_left_idx, max_d):
    # find the idx of the rightmost right locus : )
    max_left_map_value = map_values[max_left_idx]
    max_right_map_value = max_left_map_value + max_d
    max_right_idx = np.searchsorted(map_values, max_right_map_value)
    return max_right_idx


def count_site_pairs(sample_set, r_edges, window=None):
    #
    # limit defines the leftmost and rightmost left loci,
    # and the rightmost right locus is determined from the farthest map value

    if not window:
        window = [sample_set.min_position, sample_set.max_position]
    d_edges = map_util.r_to_d(r_edges)

    min_left_idx = sample_set.index_position(window[0])
    max_left_idx = sample_set.index_position(window[1]) - 1
    max_d = d_edges[-1]
    max_right_idx = get_max_right_idx(sample_set.map_values, max_left_idx, max_d)

    map_values = sample_set.map_values

    n_bins = len(r_edges) - 1
    pair_counts = np.zeros(n_bins, dtype=np.int64)

    for site_idx in np.arange(min_left_idx, max_left_idx):

        site_d_edges = map_values[site_idx] + d_edges
        pos_edges = np.searchsorted(map_values[site_idx + 1:], site_d_edges)
        pair_counts += np.diff(pos_edges)

    return pair_counts


def count_het_pairs(sample_set, sample_id, r_edges, window=None):

    if not window:
        window = [sample_set.min_position, sample_set.max_position]
    d_edges = map_util.r_to_d(r_edges)

    min_left_idx = sample_set.index_het_position(sample_id, window[0])
    max_left_idx = sample_set.index_het_position(sample_id, window[1]) - 1
    max_d = d_edges[-1]
    max_right_idx = get_max_right_idx(sample_set.get_het_map(sample_id),
                                      max_left_idx, max_d)

    het_map_values = sample_set.get_het_map(sample_id)

    n_bins = len(r_edges) - 1
    het_pair_counts = np.zeros(n_bins, dtype=np.int64)

    for site_idx in np.arange(min_left_idx, max_left_idx):

        site_d_edges = d_edges + het_map_values[site_idx]
        pos_edges = np.searchsorted(het_map_values[site_idx + 1:], site_d_edges)
        het_pair_counts += np.diff(pos_edges)

    return het_pair_counts


def count_cross_pop_het_pairs(sample_set, sample_id_x, sample_id_y, r_edges,
                              window=None):
    # there is some truly fancy stuff going on here
    d_edges = map_util.r_to_d(r_edges)
    # get those indices of sample_set.genotypes where sample X or sample Y
    # has a variant
    sample_variant_set = set(
        np.concatenate(
            (
                sample_set.get_short_variant_idx(sample_id_x),
                sample_set.get_short_variant_idx(sample_id_y)
            )
        )
    )
    variant_idx = np.sort(np.array(list(sample_variant_set), dtype=np.int64))
    if not window:
        window = [
            sample_set.variant_positions[variant_idx[0]],
            sample_set.variant_positions[variant_idx[-1]] + 1
        ]
    # these index in sample_set.genotypes
    min_left_idx = np.searchsorted(sample_set.variant_positions, window[0])
    # this is the max plus one
    max_left_idx = np.searchsorted(sample_set.variant_positions, window[1])
    max_d = d_edges[-1]
    # this is also the max plus one
    max_right_idx = get_max_right_idx(
        sample_set.variant_map_values, max_left_idx, max_d
    )
    left_site_idx = variant_idx[
        (variant_idx >= min_left_idx) & (variant_idx < max_left_idx)
    ]
    window_variant_idx = variant_idx[
        (variant_idx >= min_left_idx) & (variant_idx < max_right_idx)
    ]
    genotypes_x = sample_set.genotypes[sample_id_x][window_variant_idx]
    genotypes_y = sample_set.genotypes[sample_id_y][window_variant_idx]
    probs = compute_pr(genotypes_x, genotypes_y)
    variant_map = sample_set.variant_map_values[window_variant_idx]
    n_sites = len(left_site_idx)
    n_bins = len(r_edges) - 1
    het_pair_counts = np.zeros(n_bins, dtype=np.float64)

    for site_idx in np.arange(n_sites):
        site_pr = probs[site_idx]
        pr = site_pr * probs[site_idx + 1:]
        d_dists = variant_map[site_idx + 1:] - variant_map[site_idx]
        bin_idxs = np.searchsorted(d_edges, d_dists) - 1
        for b in np.arange(n_bins):
            het_pair_counts[b] += np.sum(pr[bin_idxs == b])

    return het_pair_counts


def count_cross_pop_het_pairs0(sample_set, sample_id_x, sample_id_y, r_edges,
                              window=None):

    if not window:
        window = [sample_set.min_position, sample_set.max_position]
    d_edges = map_util.r_to_d(r_edges)
    min_left_idx = np.searchsorted(sample_set.variant_positions, window[0])
    max_left_idx = np.searchsorted(sample_set.variant_positions, window[1]) - 1
    max_d = d_edges[-1]
    max_right_idx = get_max_right_idx(
        sample_set.variant_map_values, max_left_idx, max_d
    )
    variant_map = sample_set.variant_map_values
    genotypes_x = sample_set.genotypes[sample_id_x]
    genotypes_y = sample_set.genotypes[sample_id_y]
    probs = compute_pr(genotypes_x, genotypes_y)
    n_bins = len(r_edges) - 1
    het_pair_counts = np.zeros(n_bins, dtype=np.float64)

    for site_idx in np.arange(min_left_idx, max_left_idx):

        site_pr = probs[site_idx]
        pr = site_pr * probs[site_idx + 1:max_right_idx]
        d_dists = variant_map[site_idx + 1:max_right_idx] \
            - variant_map[site_idx]
        bin_idxs = np.searchsorted(d_edges, d_dists) - 1

        for b in np.arange(n_bins):
            het_pair_counts[b] += np.sum(pr[bin_idxs == b])

    return het_pair_counts


def compute_pr(genotypes_x, genotypes_y):

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


def save_pair_counts(pair_counts, path, r_edges):
    """
    Save a vector of pair counts

    :param pair_counts:
    :param path:
    :param r_edges:
    :return:
    """
    header = str(r_edges)
    file = open(path, 'w')
    np.savetxt(file, pair_counts, header=header)
    file.close()


def enumerate_pairs(items):
    """
    Return a list of 2-tuples containing all pairs of objects in items
    """
    n = len(items)
    pairs = []
    for i in np.arange(n):
        for j in np.arange(i + 1, n):
            pairs.append((items[i], items[j]))
    return pairs
