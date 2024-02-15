
# Functions for computing two-locus genetic statistics

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from util import sample_sets
from util import map_util
from util import one_locus


if __name__ == "__main__":
    plt.rcParams['figure.dpi'] = 100
    matplotlib.use('Qt5Agg')


r_edges = np.array([0, 1e-7, 2e-7, 5e-7, 1e-6, 2e-6, 5e-6, 1e-5, 2e-5,
                    5e-5, 1e-4, 2e-4, 5e-4, 1e-3, 2e-3, 5e-3, 1e-2, 2e-2,
                    5e-2, 1e-1],
                   dtype=np.float64)

r_mids = np.array([5e-8, 1.5e-7, 3.5e-7, 7.5e-7, 1.5e-6, 3.5e-6, 7.5e-6,
                   1.5e-5, 3.5e-5, 7.5e-5, 1.5e-4, 3.5e-4, 7.5e-4, 1.5e-3,
                   3.5e-3, 7.5e-3, 1.5e-2, 3.5e-2, 7.5e-2], dtype=np.float64)

r = r_edges[1:]


def get_last_right_idx(map_values, last_left_idx, bin_edges):
    """
    find the idx of the rightmost right locus ~ noninclusive.

    :param map_values:
    :param last_left_idx:
    :param bin_edges:
    """
    map_distance = map_util.r_to_d(bin_edges)[-1]
    left_map_val = map_values[last_left_idx - 1]  # left index noninclusive
    right_map_val = left_map_val + map_distance
    last_right_idx = np.searchsorted(map_values, right_map_val)
    return last_right_idx


def count_site_pairs(sample_set, bin_edges, window=None, limit_right=False,
                     bp_threshold=0):
    """


    :param sample_set:
    :param bin_edges:
    :param window:
    :param limit_right: if True, set the highest right locus equal to the
        highest left locus
    :param bp_threshold:
    """
    # upper window bounds are noninclusive
    if not window:
        window = sample_set.big_window
    d_edges = map_util.r_to_d(bin_edges)
    #
    first_left_idx = np.searchsorted(
        sample_set.positions, window[0], side='left'
    )
    last_left_idx = min(
        np.searchsorted(sample_set.positions, window[1]),
        sample_set.n_positions
    )
    #
    position_map = sample_set.position_map
    if limit_right:
        last_right_idx = last_left_idx
    else:
        last_right_idx = get_last_right_idx(
            position_map, last_left_idx, bin_edges
        )
    abbrev_map = position_map[first_left_idx:last_right_idx]
    abbrev_pos = sample_set.positions[first_left_idx:last_right_idx]
    #
    n_left_loci = last_left_idx - first_left_idx
    n_bins = len(bin_edges) - 1
    pair_counts = np.zeros(n_bins, dtype=np.int64)
    #
    for left_idx in np.arange(n_left_loci):
        if bp_threshold > 0:
            min_right_idx = np.searchsorted(
                    abbrev_pos, abbrev_pos[left_idx] + bp_threshold + 1
            )
        else:
            min_right_idx = left_idx + 1
        locus_edges = abbrev_map[left_idx] + d_edges
        cum_counts = np.searchsorted(abbrev_map[min_right_idx:], locus_edges)
        pair_counts += np.diff(cum_counts)
    return pair_counts


def count_het_pairs(sample_set, sample_id, bin_edges, window=None,
                    limit_right=False, bp_threshold=0):
    """


    :param sample_set:
    :param sample_id:
    :param bin_edges:
    :param window:
    :param limit_right:
    :return:
    """
    if not window:
        window = sample_set.big_window
    d_edges = map_util.r_to_d(bin_edges)
    #
    het_sites = sample_set.get_sample_het_sites(sample_id)
    first_left_idx = np.searchsorted(het_sites, window[0], side='left')
    last_left_idx = min(np.searchsorted(het_sites, window[1]), len(het_sites))
    #
    het_map = sample_set.get_sample_het_map(sample_id)
    if limit_right:
        last_right_idx = last_left_idx
    else:
        last_right_idx = get_last_right_idx(het_map, last_left_idx, bin_edges)
    abbrev_map = het_map[first_left_idx:last_right_idx]
    abbrev_pos = het_sites[first_left_idx:last_right_idx]
    #
    n_left_loci = last_left_idx - first_left_idx
    n_bins = len(bin_edges) - 1
    het_pair_counts = np.zeros(n_bins, dtype=np.int64)
    # the loop
    for left_idx in np.arange(n_left_loci):
        if bp_threshold > 0:
            min_right_idx = np.searchsorted(
                    abbrev_pos, abbrev_pos[left_idx] + bp_threshold + 1
            )
        else:
            min_right_idx = left_idx + 1
        locus_edges = abbrev_map[left_idx] + d_edges
        cum_counts = np.searchsorted(abbrev_map[min_right_idx:], locus_edges)
        het_pair_counts += np.diff(cum_counts)
    return het_pair_counts


def count_two_sample_het_pairs(sample_set, sample_id_x, sample_id_y, bin_edges,
                               window=None, limit_right=False,
                               bp_threshold=0):
    """


    :param sample_set:
    :param sample_id_x:
    :param sample_id_y:
    :param bin_edges:
    :param window:
    """
    if not window:
        window = sample_set.big_window
    d_edges = map_util.r_to_d(bin_edges)
    # find the indices for the first and last left loci
    joint_var_idx = sample_set.get_multi_sample_variant_idx(
        sample_id_x, sample_id_y
    )
    first_left_idx = np.searchsorted(
        sample_set.variant_sites[joint_var_idx], window[0], side='left'
    )
    last_left_idx = min(
        np.searchsorted(sample_set.variant_sites[joint_var_idx], window[1]),
        len(joint_var_idx)
    )
    # find the index of the last accessible right locus
    if limit_right:
        last_right_idx = last_left_idx
    else:
        last_right_idx = get_last_right_idx(
            sample_set.variant_site_map[joint_var_idx], last_left_idx,
            bin_edges
        )
    # slice the joint variant index
    windowed_idx = joint_var_idx[first_left_idx:last_right_idx]
    #
    genotypes_x = sample_set.genotypes[sample_id_x][windowed_idx]
    genotypes_y = sample_set.genotypes[sample_id_y][windowed_idx]
    diff_probs = one_locus.compute_site_diff_probs(genotypes_x, genotypes_y)
    variant_map = sample_set.variant_site_map[windowed_idx]
    abbrev_pos = sample_set.variant_sites[windowed_idx]
    #
    n_left_loci = last_left_idx - first_left_idx
    n_bins = len(bin_edges) - 1
    het_pair_counts = np.zeros(n_bins, dtype=np.float64)
    # loop over left loci and compute two-locus heterozygosity
    for left_idx in np.arange(n_left_loci):
        if bp_threshold > 0:
            min_right_idx = np.searchsorted(
                    abbrev_pos, abbrev_pos[left_idx] + bp_threshold + 1
            )
        else:
            min_right_idx = left_idx + 1
        site_diff_prob = diff_probs[left_idx]
        joint_diff_probs = site_diff_prob * diff_probs[min_right_idx:]
        #
        d_dists = variant_map[min_right_idx:] - variant_map[left_idx]
        edges = np.searchsorted(d_dists, d_edges)
        #
        for b in np.arange(n_bins):
            het_pair_counts[b] += np.sum(joint_diff_probs[edges[b]:edges[b+1]])
        #
    return het_pair_counts
