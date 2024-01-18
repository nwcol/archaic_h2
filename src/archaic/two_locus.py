import matplotlib.pyplot as plt

import numpy as np

import sys

sys.path.append("/home/nick/Projects/archaic/src")

import archaic.vcf_samples as vcf_samples

import archaic.map_util as map_util


r_edges = np.array([0,
                    1e-7, 2e-7, 5e-7,
                    1e-6, 2e-6, 5e-6,
                    1e-5, 2e-5, 5e-5,
                    1e-4, 2e-4, 5e-4,
                    1e-3, 2e-3, 5e-3,
                    1e-2, 2e-2, 5e-2,
                    1e-1, 2e-1, 5e-1], dtype=np.float64)
r = r_edges[1:]


def bin_pairs(samples, r_edges=r_edges):
    """
    Get the number of site pairs per bin using bins specified by r_edges

    :param samples:
    :param r_edges:
    :return:
    """
    n_bins = len(r_edges) - 1
    n_positions = samples.n_positions
    pair_counts = np.zeros(n_bins, dtype=np.int64)
    d_edges = map_util.r_to_d(r_edges)  # convert bins in r to bins in d
    # d_edges[0] -= 1e-4
    map_values = samples.map_values
    i = 0
    for i in np.arange(n_positions):
        pos_value = map_values[i]
        edges_for_i = d_edges + pos_value
        pos_edges = np.searchsorted(map_values[i+1:], edges_for_i)
        pair_counts += np.diff(pos_edges)
        if i % 1e6 == 0:
            print(f"{i + 1} bp scanned, {np.sum(pair_counts)} pairs binned")
    print(f"{i + 1} bp scanned, {np.sum(pair_counts)} pairs binned")
    exp = int(n_positions * (n_positions - 1) / 2)
    emp = int(np.sum(pair_counts))
    diff = emp - exp
    print(f"{emp} pairs detected for {i+1} sites, expected {exp}, "
          f"difference {diff}")
    return pair_counts


def bin_het_pairs(samples, sample_id, r_edges=r_edges):
    """

    :param samples:
    :param sample_id:
    :param r_edges:
    :return:
    """
    n_bins = len(r_edges) - 1
    n_het = samples.n_het(sample_id)
    pair_counts = np.zeros(n_bins, dtype=np.int64)
    d_edges = map_util.r_to_d(r_edges)  # convert bins in r to bins in d
    # d_edges[0] -= 1e-4  # ensure that the lowest bin is counted properly
    het_map = samples.map_values[samples.het_index(sample_id)]
    i = 0
    for i in np.arange(n_het):
        value = het_map[i]
        edges_for_i = d_edges + value
        pos_edges = np.searchsorted(het_map[i+1:], edges_for_i)
        pair_counts += np.diff(pos_edges)
    exp = int(n_het * (n_het - 1) / 2)
    emp = int(np.sum(pair_counts))
    diff = emp - exp
    print(f"{emp} pairs / {exp} expected for {i+1} sites, "
          f"difference {diff}")
    return pair_counts


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


def compute_pi_2(samples, sample_id, pair_counts, r_edges=r_edges):
    """
    Compute joint heterozygosity for one sample in a Samples instance

    :param samples:
    :param sample_id:
    :param pair_counts:
    :param r_edges:
    :return:
    """
    het_pairs = bin_het_pairs(samples, sample_id, r_edges=r_edges)
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
        het_pairs[sample_id] = bin_het_pairs(samples, sample_id,
                                              r_edges=r_edges)
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


# two sample joint heterozygosity


def haplotype_probs(sample):
    """
    Return [P(AB), P(Ab), P(aB), P(ab)]
    Given a 2-tuple/vector/list of alternate allele counts for an unphased
    sample, compute the probabilities of sampling the possible phased
    haplotypes.

    :param sample: vector of length 2; alternate allele counts for 2 sites
    :return:
    """
    f_A = sample[0] / 2
    f_B = sample[1] / 2
    f_a = 1 - f_A
    f_b = 1 - f_B
    probs = np.array([f_A * f_B, f_A * f_b, f_a * f_B, f_a * f_b],
                     dtype=np.float64)
    return probs


def joint_het_test(sample_0, sample_1):
    """
    For two 2-tuples/vectors/lists of alternate allele counts at two sites in
    two samples, compute the joint heterozygosity.

    The joint heterozygosity for unphased samples equals
    P_0(AB)P_1(ab) + P_0(Ab)P_1(bA) + P_0(aB)P_1(Ab) + P_0(ab)P_1(AB)

    :param sample_0:
    :param sample_1:
    :return:
    """
    hap_probs_0 = haplotype_probs(sample_0)
    hap_probs_1 = haplotype_probs(sample_1)
    hets = hap_probs_0 * np.flip(hap_probs_1)
    joint_het = np.sum(hets)
    return joint_het


def joint_het_arrs(sample_0, sample_1):

    n_positions = len(sample_0)
    n_pairs = int(0.5 * n_positions * (n_positions - 1))
    joint_hets = np.zeros(n_pairs)
    tot = 0

    for i in np.arange(n_positions):

        for j in np.arange(i + 1, n_positions):

            joint_hets[tot] = joint_het_test(sample_0[[i, j]], sample_1[[i, j]])
            tot += 1

    joint_het = np.sum(joint_hets) / tot

    return joint_hets, tot, joint_het


def fast_haplotype_probs(alt_counts):
    """
    np.array([#A, #B])

    :param alt_counts:
    :return:
    """
    alt_freqs = alt_counts / 2
    prob_matrix = np.outer(alt_freqs, 1 - alt_freqs)
    return prob_matrix


def get_sample_hap_arr(samples, sample_id):
    """
    return an array of haplotype sample probabilities for unphased samples

    capital letters = alt alleles

    :param sample:
    :return:
    """
    alts = samples.samples[sample_id] / 2
    refs = 1 - alts
    A_freq = 0
    B_freq = 1
    a_freq = 1 - A_freq
    b_freq = 1 - B_freq
    probs = np.array([f_A * f_B, f_A * f_b, f_a * f_B, f_a * f_b],
                     dtype=np.float64)
    return 0


chr22_samples = vcf_samples.UnphasedSamples.dir(
    "/home/nick/Projects/archaic/data/chromosomes/merged/chr22/")
