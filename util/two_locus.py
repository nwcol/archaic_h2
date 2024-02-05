
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
                       3.5e-3, 7.5e-3, 1.5e-2, 3.5e-2, 7.5e-2, 1.5e-1, 3.5e-1],
                      dtype=np.float64)

r_bin_right_edges = r_bin_edges[1:]




ex = sample_sets.UnphasedSampleSet.get_chr(22)


def get_max_right_idx(map_values, max_left_idx, max_d):
    # find the idx of the rightmost right locus : )
    max_left_map_value = map_values[max_left_idx]
    max_right_map_value = max_left_map_value + max_d
    max_right_idx = np.searchsorted(map_values, max_right_map_value)
    return max_right_idx


def count_site_pairs(sample_set, r_edges, limits=None):
    #
    # limit defines the leftmost and rightmost left loci,
    # and the rightmost right locus is determined from the farthest map value

    if not limits:
        limits = [sample_set.min_position, sample_set.max_position]
    d_edges = map_util.r_to_d(r_edges)

    min_left_idx = sample_set.index_position(limits[0])
    max_left_idx = sample_set.index_position(limits[1]) - 1
    max_d = d_edges[-1]
    max_right_idx = get_max_right_idx(sample_set.map_values, max_left_idx, max_d)

    map_values = sample_set.map_values[min_left_idx:max_right_idx]

    n_bins = len(r_edges) - 1
    pair_counts = np.zeros(n_bins, dtype=np.int64)

    for site_idx in np.arange(min_left_idx, max_left_idx):

        site_d_edges = map_values[site_idx] + d_edges
        pos_edges = np.searchsorted(map_values[site_idx + 1:], site_d_edges)
        pair_counts += np.diff(pos_edges)

    return pair_counts


def count_het_pairs(sample_set, sample_id, r_edges, limits=None):

    if not limits:
        limits = [sample_set.min_position, sample_set.max_position]
    d_edges = map_util.r_to_d(r_edges)

    min_left_idx = sample_set.index_het_position(sample_id, limits[0])
    max_left_idx = sample_set.index_het_position(sample_id, limits[1]) - 1
    max_d = d_edges[-1]
    max_right_idx = get_max_right_idx(sample_set.get_het_map(sample_id),
                                      max_left_idx, max_d)

    het_map_values = sample_set.get_het_map(sample_id)[min_left_idx:max_right_idx]

    n_bins = len(r_edges) - 1
    het_pair_counts = np.zeros(n_bins, dtype=np.int64)

    for site_idx in np.arange(min_left_idx, max_left_idx):

        site_d_edges = d_edges + het_map_values[site_idx]
        pos_edges = np.searchsorted(het_map_values[site_idx + 1:], site_d_edges)
        het_pair_counts += np.diff(pos_edges)

    return het_pair_counts






def c0unt_het_pairs(sample_set, sample_id, r_edges):

    d_edges = map_util.r_to_d(r_edges)
    n_het = sample_set.n_het(sample_id)
    het_map = sample_set.get_het_map(sample_id)

    n_bins = len(r_edges) - 1
    het_pair_counts = np.zeros(n_bins, dtype=np.int64)
    for i in np.arange(n_het):
        edges_for_i = d_edges + het_map[i]
        pos_edges = np.searchsorted(het_map[i + 1:], edges_for_i)
        het_pair_counts += np.diff(pos_edges)
    return het_pair_counts


def c0unt_site_pairs(sample_set, r_edges):


    n_positions = sample_set.n_positions
    map_values = sample_set.map_values

    n_bins = len(r_edges) - 1
    pair_counts = np.zeros(n_bins, dtype=np.int64)
    for i in np.arange(n_positions):
        edges_for_i = d_edges + map_values[i]
        pos_edges = np.searchsorted(map_values[i + 1:], edges_for_i)
        pair_counts += np.diff(pos_edges)
        if i % 1e6 == 0 and i > 0:
            print(f"{i + 1} bp scanned, {np.sum(pair_counts)} pairs binned")
    return pair_counts


def count_cross_pop_het_pairs0(sample_set, sample_id_x, sample_id_y, r_edges,
                              limits=None):

    d_edges = map_util.r_to_d(r_edges)

    # get those sites where sample X or sample Y has a variant
    variant_idx_set = set(
        np.concatenate(
            (
                sample_set.get_short_variant_idx(sample_id_x),
                sample_set.get_short_variant_idx(sample_id_y)
            )
        )
    )
    # indexes positions
    variant_idx = np.sort(np.array(list(variant_idx_set), dtype=np.int64))

    # limits; inclusive?
    if not limits:
        limits = [
            sample_set.variant_positions[variant_idx[0]],
            sample_set.variant_positions[variant_idx[-1]] + 1
        ]

    # these index "variant_positions"
    subset_positions = sample_set.variant_positions[variant_idx]
    min_left_idx = np.searchsorted(subset_positions, limits[0])
    max_left_idx = np.searchsorted(subset_positions, limits[1]) - 1

    print(min_left_idx, max_left_idx)

    max_d = d_edges[-1]
    max_right_idx = get_max_right_idx(
        sample_set.variant_map_values[variant_idx], max_left_idx, max_d
    )

    print(max_right_idx)

    variant_map = sample_set.variant_map_values[variant_idx]

    genotypes_x = sample_set.genotypes[sample_id_x][variant_idx]
    genotypes_y = sample_set.genotypes[sample_id_y][variant_idx]

    probs = compute_pr(genotypes_x, genotypes_y)

    n_bins = len(r_edges) - 1
    het_pair_counts = np.zeros(n_bins, dtype=np.float64)

    # site_idx indexes variant_map_values
    for site_idx in np.arange(min_left_idx, max_left_idx):

        # max_concern = np.searchsorted(bin_idxs, n_bins)  # out of range of
        # highest bin!!!

        site_pr = probs[site_idx]
        pr = site_pr * probs[site_idx + 1:]

        d_dists = variant_map[site_idx + 1:] - variant_map[site_idx]
        bin_idxs = np.searchsorted(d_edges, d_dists) - 1

        for b in np.arange(n_bins):
            het_pair_counts[b] += np.sum(pr[bin_idxs == b])

    return het_pair_counts



def count_cross_pop_het_pairs(sample_set, sample_id_x, sample_id_y, r_edges,
                              limits=None):

    if not limits:
        limits = [sample_set.min_position, sample_set.max_position]
    d_edges = map_util.r_to_d(r_edges)

    min_left_idx = np.searchsorted(sample_set.variant_positions, limits[0])
    max_left_idx = np.searchsorted(sample_set.variant_positions, limits[1]) - 1
    print(min_left_idx, max_left_idx)

    max_d = d_edges[-1]
    max_right_idx = get_max_right_idx(
        sample_set.variant_map_values, max_left_idx, max_d
    )
    print(max_right_idx)

    variant_map = sample_set.variant_map_values
    genotypes_x = sample_set.genotypes[sample_id_x]
    genotypes_y = sample_set.genotypes[sample_id_y]
    probs = compute_pr(genotypes_x, genotypes_y)

    n_bins = len(r_edges) - 1
    het_pair_counts = np.zeros(n_bins, dtype=np.float64)

    # site_idx indexes variant_map_values
    for site_idx in np.arange(min_left_idx, max_left_idx):

        # max_concern = np.searchsorted(bin_idxs, n_bins)  # out of range of
        # highest bin!!!

        site_pr = probs[site_idx]
        pr = site_pr * probs[site_idx + 1:]

        d_dists = variant_map[site_idx + 1:] - variant_map[site_idx]
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



def test_new(gen_x, gen_y):


    pr = compute_pr(gen_x, gen_y)

    return pr[0] * pr[1:]





def test_old(gen_x, gen_y):

    freq_x = np.sum(gen_x, axis=1)/ 2
    freq_y = np.sum(gen_y, axis=1) / 2

    pr_x = get_haplotype_prob_arr(freq_x[0], freq_x[1:])
    pr_y = get_haplotype_prob_arr(freq_y[0], freq_y[1:])
    hets = compute_hets(pr_x, pr_y)

    return hets


def new_bin(i, mapp, r_edges):

    d_edges = map_util.r_to_d(r_edges)
    d_dists = mapp[i + 1:] - mapp[i]
    bin_idxs = np.searchsorted(d_edges, d_dists)
    return bin_idxs

def assign_bins(i, alt_map, r_edges):
    """
    Assign the positions mapped by alt_map above index i to recombination
    distance bins given by r_edges and return a vector of indices

    :param i: index to which r values are relative
    :param alt_map: vector of map values
    :param r_edges: vector of r bin edges
    :return:
    """
    d_dist = alt_map[i + 1:] - alt_map[i]
    r_dist = map_util.d_to_r(d_dist)
    return np.searchsorted(r_edges, r_dist) - 1


def c0unt_cross_pop_het_pairs(samples, sample_id_x, sample_id_y,
                                  r_edges):

    alt_map = samples.alt_map_values
    freq_x = (samples.samples[sample_id_x] / 2).astype(np.float64)
    freq_y = (samples.samples[sample_id_y] / 2).astype(np.float64)
    n_alts = samples.n_variants
    n_bins = len(r_edges) - 1
    joint_het = np.zeros(n_bins)

    for i in np.arange(n_alts):

        if freq_x[i] != 0 or freq_y[i] != 0:

            pr_x = get_haplotype_prob_arr(freq_x[i], freq_x[i + 1:])
            pr_y = get_haplotype_prob_arr(freq_y[i], freq_y[i + 1:])
            hets = compute_hets(pr_x, pr_y)
            bin_idx = assign_bins(i, alt_map, r_edges)

            for b in np.arange(n_bins):
                joint_het[b] += np.sum(hets[bin_idx == b])

        if i % 10_000 == 0 and i > 0:
            print(i)

    return joint_het





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


def get_haplotype_prob_arr(A, B_vec):
    """
    For unphased samples; given the frequencies of alternate alleles, compute
    the probabilities that the haplotypes AB, Ab, aB, ab would be sampled from
    a phased sample.

    Operates on a single focal locus with alleles named a, A and a vector of
    other loci named the B_vec.

    :param A: alternate allele frequency at a site of interest
    :type A: float
    :param B_vec: vector of alternate allele frequencies outside side of
        interest
    :type B_vec: 1d array of np.float64
    :return: 2d array of with probabilities arrayed in columns, shape (4, n)
    """
    a = 1 - A
    b_vec = 1 - B_vec
    probs = np.array([A * B_vec, A * b_vec, a * B_vec, a * b_vec],
                     dtype=np.float64)
    return probs


def compute_hets(pr_x, pr_y):
    """
    Compute the probability of sampling doubly heterozygous haplotypes in
    both populations x and y

    Operates on arrays of haplotype sample probabilities; probabilities are
    used because our samples are unphased.

    :param pr_x: haplotype sample probs for population x
    :param pr_y: haplotype sample probs for pop y
    :return:
    """
    pr_y = np.flip(pr_y, axis=0)
    return np.sum(pr_x * pr_y, axis=0)

















def compute_pi_2(samples, sample_id, pair_counts, r_edges):
    """
    Compute joint heterozygosity for one sample in a Samples instance

    :param samples:
    :param sample_id:
    :param pair_counts:
    :param r_edges:
    :return:
    """
    het_pairs = count_heterozygous_pairs(samples, sample_id, r_edges=r_edges)
    pi_2 = het_pairs / pair_counts
    return pi_2


def compute_pi_2s(samples, pair_counts, r_edges):
    """
    Compute joint heterozygosity for each sample in a Samples instance, given
    the vector of pair counts.

    :param samples:
    :param pair_counts:
    :param r_edges:
    :return:
    """
    pi_2_dict = {sample_id: None for sample_id in samples.sample_ids}
    for sample_id in pi_2_dict:
        pi_2_dict[sample_id] = compute_pi_2(samples, sample_id, pair_counts,
                                            r_edges=r_edges)
    return pi_2_dict


def get_het_pairs(samples, r_edges):
    """
    Get a dictionary of heterozygous pair counts for bins given by r_edges

    :param samples:
    :param r_edges:
    :return:
    """
    het_pairs = {sample_id: None for sample_id in samples.sample_ids}
    for sample_id in het_pairs:
        het_pairs[sample_id] = count_heterozygous_pairs(samples, sample_id)
    return het_pairs


def make_het_pair_matrix(het_pair_dict):
    n_samples = len(het_pair_dict)
    n_bins = len(list(het_pair_dict.values())[0])
    sample_ids = []
    pair_matrix = np.zeros((n_samples, n_bins), dtype=np.float64)
    for i, sample_id in enumerate(het_pair_dict):
        pair_matrix[i, :] = het_pair_dict[sample_id]
        sample_ids.append(sample_id)
    return pair_matrix, sample_ids


def het_matrix_to_dict(pair_matrix, sample_ids):
    het_dict = {sample_id: None for sample_id in sample_ids}
    for i, sample_id in enumerate(sample_ids):
        het_dict[sample_id] = pair_matrix[i]
    return het_dict


def save_het_pairs(het_pairs, path):

    n_samples = len(het_pairs)
    n_bins = len(list(het_pairs.values())[0])
    sample_ids = []
    pair_matrix = np.zeros((n_samples, n_bins), dtype=np.float64)
    for i, sample_id in enumerate(het_pairs):
        pair_matrix[i, :] = het_pairs[sample_id]
        sample_ids.append(sample_id)
    file = open(path, 'w')
    np.savetxt(file, pair_matrix, header=str(sample_ids))
    file.close()
    return 0


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
