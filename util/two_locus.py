
"""
Functions for computing two-locus genetic statistics
"""

import numpy as np
from util import map_util


def count_site_pairs(sample_set, bin_edges, window=None, limit_right=False,
                     bp_threshold=0):
    """

    """
    if not window:
        window = sample_set.big_window
    first_left_idx, last_left_idx = get_window_idxs(
        sample_set.positions, window
    )
    if limit_right:
        last_right_idx = last_left_idx
    else:
        last_right_idx = get_last_right_idx(
            sample_set.position_map, last_left_idx, bin_edges
        )
    d_edges = map_util.r_to_d(bin_edges)
    abbrev_map = sample_set.position_map[first_left_idx:last_right_idx]
    abbrev_pos = sample_set.positions[first_left_idx:last_right_idx]
    n_left_loci = last_left_idx - first_left_idx
    cum_counts = np.zeros(len(bin_edges), dtype=np.int64)
    #
    for left_idx in np.arange(n_left_loci):
        if bp_threshold > 0:
            min_right_idx = np.searchsorted(
                    abbrev_pos, abbrev_pos[left_idx] + bp_threshold + 1
            )
        else:
            min_right_idx = left_idx + 1
        locus_edges = d_edges + abbrev_map[left_idx]
        cum_counts += np.searchsorted(abbrev_map[min_right_idx:], locus_edges)
    #
    pair_counts = np.diff(cum_counts)
    return pair_counts


def count_het_pairs(sample_set, sample_id, bin_edges, window=None,
                    limit_right=False, bp_threshold=0):
    """
    Count the number of jointly heterozygous site pairs w for a single sample
    in a series of bins given in units of r.
    """
    if not window:
        window = sample_set.big_window
    d_edges = map_util.r_to_d(bin_edges)
    het_sites = sample_set.het_sites(sample_id)
    first_left_idx, last_left_idx = get_window_idxs(het_sites, window)
    het_map = sample_set.het_map(sample_id)
    if limit_right:
        last_right_idx = last_left_idx
    else:
        last_right_idx = get_last_right_idx(
            het_map, last_left_idx, bin_edges
        )
    abbrev_map = het_map[first_left_idx:last_right_idx]
    abbrev_pos = het_sites[first_left_idx:last_right_idx]
    n_left_loci = last_left_idx - first_left_idx
    cum_counts = np.zeros(len(bin_edges), dtype=np.int64)
    #
    for left_idx in np.arange(n_left_loci):
        if bp_threshold > 0:
            min_right_idx = np.searchsorted(
                    abbrev_pos, abbrev_pos[left_idx] + bp_threshold + 1
            )
        else:
            min_right_idx = left_idx + 1
        site_d_edges = d_edges + abbrev_map[left_idx]
        cum_counts += np.searchsorted(abbrev_map[min_right_idx:], site_d_edges)
    #
    het_pair_counts = np.diff(cum_counts)
    return het_pair_counts


def count_per_site_het_pairs(sample_set, sample_id, bin_edges, window=None,
                             limit_right=False, bp_threshold=0):
    if not window:
        window = sample_set.big_window
    d_edges = map_util.r_to_d(bin_edges)
    het_sites = sample_set.het_sites(sample_id)
    first_left_idx, last_left_idx = get_window_idxs(het_sites, window)
    het_map = sample_set.het_map(sample_id)
    if limit_right:
        last_right_idx = last_left_idx
    else:
        last_right_idx = get_last_right_idx(
            het_map, last_left_idx, bin_edges
        )
    abbrev_map = het_map[first_left_idx:last_right_idx]
    abbrev_pos = het_sites[first_left_idx:last_right_idx]
    n_left_loci = last_left_idx - first_left_idx
    out = []
    #
    for left_idx in np.arange(n_left_loci):
        if bp_threshold > 0:
            min_right_idx = np.searchsorted(
                    abbrev_pos, abbrev_pos[left_idx] + bp_threshold + 1
            )
        else:
            min_right_idx = left_idx + 1
        site_d_edges = d_edges + abbrev_map[left_idx]
        x = np.searchsorted(abbrev_map[min_right_idx:], site_d_edges)
        out.append(np.diff(x))
    #
    return np.array(out)


def count_per_site_site_pairs(sample_set, bin_edges, window=None,
                              limit_right=False, bp_threshold=0):
    if not window:
        window = sample_set.big_window
    first_left_idx, last_left_idx = get_window_idxs(
        sample_set.positions, window
    )
    if limit_right:
        last_right_idx = last_left_idx
    else:
        last_right_idx = get_last_right_idx(
            sample_set.position_map, last_left_idx, bin_edges
        )
    d_edges = map_util.r_to_d(bin_edges)
    abbrev_map = sample_set.position_map[first_left_idx:last_right_idx]
    abbrev_pos = sample_set.positions[first_left_idx:last_right_idx]
    n_left_loci = last_left_idx - first_left_idx
    out = []
    #
    for left_idx in np.arange(n_left_loci):
        if bp_threshold > 0:
            min_right_idx = np.searchsorted(
                    abbrev_pos, abbrev_pos[left_idx] + bp_threshold + 1
            )
        else:
            min_right_idx = left_idx + 1
        locus_edges = d_edges + abbrev_map[left_idx]
        out.append(np.searchsorted(abbrev_map[min_right_idx:], locus_edges))
    #
    out = np.array(out)
    return np.diff(out, axis=1)


def count_two_sample_het_pairs(sample_set, sample_id_x, sample_id_y, bin_edges,
                               window=None, limit_right=False):
    """
    Compute the sums of probabilities of sampling distinct alleles at both
    sites in site pairs between two samples, in bins given in units of r.

    H_2_XY = Pr(distinct left alleles) * Pr(distinct right alleles)
    """
    if not window:
        window = sample_set.big_window
    var_idx = sample_set.multi_sample_variant_idx(sample_id_x, sample_id_y)
    first_left_idx, last_left_idx = get_window_idxs(
        sample_set.variant_sites[var_idx], window
    )
    if limit_right:
        last_right_idx = last_left_idx
    else:
        last_right_idx = get_last_right_idx(
            sample_set.variant_site_map[var_idx], last_left_idx, bin_edges
        )
    slic = slice(first_left_idx, last_right_idx)
    abbrev_map = sample_set.variant_site_map[var_idx][slic]
    site_diff_probs = \
        sample_set.site_diff_probs(sample_id_x, sample_id_y)[var_idx][slic]
    site_diff_probs = np.append(site_diff_probs, 0)
    precomputed = {
        x: np.cumsum(x * site_diff_probs) for x in [0., 0.25, 0.5, 0.75, 1.]
    }
    d_edges = map_util.r_to_d(bin_edges)
    right_lims = np.searchsorted(abbrev_map, abbrev_map + d_edges[-1])
    cum_counts = np.zeros(len(bin_edges), dtype=np.float64)
    n_left_sites = last_left_idx - first_left_idx
    #
    for left_idx in np.arange(n_left_sites):
        lim = right_lims[left_idx]
        site_diff_prob = site_diff_probs[left_idx]
        joint_diff_probs = precomputed[site_diff_prob][left_idx:lim]
        site_d_edges = d_edges + abbrev_map[left_idx]
        edges = np.searchsorted(abbrev_map[left_idx + 1:lim], site_d_edges)
        cum_counts += joint_diff_probs[edges]
    #
    het_pair_counts = np.diff(cum_counts)
    return het_pair_counts


def get_window_idxs(pos, window):
    """
    Return the indices that access the first and last positions in a
    given window. Upper position is noninclusive.
    """
    lower, upper = window
    first_idx = np.searchsorted(pos, lower)
    last_idx = np.searchsorted(pos, upper)
    return first_idx, last_idx


def get_last_right_idx(map_vec, last_left_idx, bin_edges):
    """
    Find the index of the rightmost right locus ~ noninclusive.
    """
    map_distance = map_util.r_to_d(bin_edges)[-1]
    left_map_val = map_vec[last_left_idx - 1]
    right_map_val = left_map_val + map_distance
    last_right_idx = np.searchsorted(map_vec, right_map_val)
    return last_right_idx


r_edges = np.array([0,
                    1e-7, 2e-7, 5e-7,
                    1e-6, 2e-6, 5e-6,
                    1e-5, 2e-5, 5e-5,
                    1e-4, 2e-4, 5e-4,
                    1e-3, 2e-3, 5e-3,
                    1e-2, 2e-2, 5e-2,
                    1e-1], dtype=np.float64)

r = r_edges[1:]

short_r_edges = np.array([0,
                          1e-7, 2e-7, 5e-7,
                          1e-6, 2e-6, 5e-6,
                          1e-5], dtype=np.float64)
