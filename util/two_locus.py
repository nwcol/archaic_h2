
#

import matplotlib

import matplotlib.pyplot as plt

import numpy as np

import sys

import time

from util import vcf_samples

from util import map_util


if __name__ == "__main__":
    plt.rcParams['figure.dpi'] = 100
    matplotlib.use('Qt5Agg')


r_edges = np.array([0,
                   1e-7, 2e-7, 5e-7,
                   1e-6, 2e-6, 5e-6,
                   1e-5, 2e-5, 5e-5,
                   1e-4, 2e-4, 5e-4,
                   1e-3, 2e-3, 5e-3,
                   1e-2, 2e-2, 5e-2,
                   1e-1, 2e-1, 5e-1], dtype=np.float64)

r_mids = np.array([5.0e-8, 1.5e-7, 3.5e-7,
                   7.5e-7, 1.5e-6, 3.5e-6,
                   7.5e-6, 1.5e-5, 3.5e-5,
                   7.5e-5, 1.5e-4, 3.5e-4,
                   7.5e-4, 1.5e-3, 3.5e-3,
                   7.5e-3, 1.5e-2, 3.5e-2,
                   7.5e-2, 1.5e-1, 3.5e-1])

r = r_edges[1:]


def count_site_pairs(samples, r_edges=r_edges):
    """
    Get the number of site pairs per bin using bins specified by r_edges

    :param samples:
    :param r_edges:
    :return:
    """
    n_bins = len(r_edges) - 1
    n_positions = samples.n_positions
    pair_counts = np.zeros(n_bins, dtype=np.int64)
    d_edges = map_util.r_to_d(r_edges)
    map_values = samples.map_values
    for i in np.arange(n_positions):
        pos_value = map_values[i]
        edges_for_i = d_edges + pos_value
        pos_edges = np.searchsorted(map_values[i+1:], edges_for_i)
        pair_counts += np.diff(pos_edges)
        if i % 1e6 == 0:
            print(f"{i + 1} bp scanned, {np.sum(pair_counts)} pairs binned")
    print(f"{i + 1} bp scanned, {np.sum(pair_counts)} pairs binned")
    exp = int(n_positions * (n_positions - 1) / 2)
    emp = np.sum(pair_counts)
    print(f"{emp} pairs detected for {i+1} sites, expected {exp}, "
          f"difference {emp - exp}")
    return pair_counts


def count_heterozygous_pairs(samples, sample_id, r_edges=r_edges):
    """

    :param samples:
    :param sample_id:
    :param r_edges:
    :return:
    """
    n_bins = len(r_edges) - 1
    n_het = samples.n_het(sample_id)
    pair_counts = np.zeros(n_bins, dtype=np.int64)
    d_edges = map_util.r_to_d(r_edges)
    het_map = samples.map_values[samples.het_index(sample_id)]
    for i in np.arange(n_het):
        value = het_map[i]
        edges_for_i = d_edges + value
        pos_edges = np.searchsorted(het_map[i+1:], edges_for_i)
        pair_counts += np.diff(pos_edges)
    exp = int(n_het * (n_het - 1) / 2)
    emp = np.sum(pair_counts)
    print(f"{emp} pairs / {exp} expected for {i+1} sites, "
          f"difference {emp - exp}")
    return pair_counts


def count_cross_pop_heterozygous_pairs(samples, sample_id_x, sample_id_y,
                                       r_edges=r_edges):
    """

    :param samples:
    :param sample_id_x:
    :param sample_id_y:
    :param r_edges:
    :return:
    """
    time_0 = time.time()
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
            print(i, np.round(time.time() - time_0, 2), " s")

    return joint_het


def save_pair_counts(pair_counts, path, r_edges=r_edges):
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
















def compute_pi_2(samples, sample_id, pair_counts, r_edges=r_edges):
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


def compute_pi_2s(samples, pair_counts, r_edges=r_edges):
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


def get_het_pairs(samples, r_edges=r_edges):
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


def plot_pi_2_dict(pi_2_dict, r=r):

    fig = plt.figure(figsize=(8, 6))
    sub = fig.add_subplot(111)
    for sample_id in pi_2_dict:
        plt.plot(r, pi_2_dict[sample_id], marker='x', label=sample_id)
    sub.set_xscale("log")
    sub.set_xlabel("r bin")
    sub.set_ylabel("pi_2")
    sub.set_ylim(0,)
    sub.legend()
    plt.tight_layout()
    fig.show()
    return 0


def enumerate_pairs(items):
    """
    Return a list of 2-tuples containing all pairs of objects in items

    :param items: list of objects to pair
    :return:
    """
    n = len(items)
    pairs = []
    for i in np.arange(n):
        for j in np.arange(i + 1, n):
            pairs.append((items[i], items[j]))
    return pairs

"""
samples = {x: vcf_samples.UnphasedSamples.dir(
    f"/home/nick/Projects/archaic/data/chromosomes/merged/chr{x}/")
    for x in np.arange(13, 23)}

pair_counts = {x: np.loadtxt(
    f"/home/nick/Projects/archaic/data/statistics/pair_counts/chr{x}"
    "_pair_counts.txt")
    for x in np.arange(13, 23)}

"""
