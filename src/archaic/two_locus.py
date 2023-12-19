import matplotlib.pyplot as plt

import matplotlib

import numpy as np

import time

import os

import sys

import samples


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
r = r_edges[1:]



def get_pair_distribution(masked_map, r_edges):
    """
    Get the number of site pairs per bin for an array of bins in r

    :param genotype_vector:
    :param masked_map:
    :param r_bins:
    :param max_i:
    :return:
    """
    n_bins = len(r_edges) - 1
    n_pos = masked_map.length
    pair_counts = np.zeros(n_bins, dtype=np.int64)
    d_edges = map_util.r_to_d(r_edges)  # convert bins in r to bins in d
    d_edges[0] -= 1e-4
    values = masked_map.values
    i = 0
    for i in np.arange(n_pos):
        pos_value = values[i]
        edges_for_i = d_edges + pos_value
        pos_edges = np.searchsorted(values[i+1:], edges_for_i)
        pair_counts += np.diff(pos_edges)
        if i % 1e6 == 0:
            print(f"{i} bp scanned, {np.sum(pair_counts)} pairs binned")
    print(f"{i} bp scanned, {np.sum(pair_counts)} pairs binned")
    expected_n_pairs = int(n_pos * (n_pos - 1) / 2)
    n_pairs = int(np.sum(pair_counts))
    diff = int(n_pairs - expected_n_pairs)
    print(f"{n_pairs} recorded out of {expected_n_pairs}, difference {diff}")
    return pair_counts


def get_het_pair_distribution(genotype_vector, masked_map, r_edges):
    """
    Get the number of site pairs per bin for an array of bins in r

    :param genotype_vector:
    :param masked_map:
    :param r_bins:
    :param max_i:
    :return:
    """
    n_bins = len(r_edges) - 1
    n_het = genotype_vector.n_het
    pair_counts = np.zeros(n_bins, dtype=np.int64)
    d_edges = map_util.r_to_d(r_edges)  # convert bins in r to bins in d
    d_edges[0] -= 1e-4  # ensure that the lowest bin is counted properly
    # without this, the first few values above i may be missed
    het_map = masked_map.values[genotype_vector.het_index]
    i = 0
    for i in np.arange(n_het):
        value = het_map[i]
        edges_for_i = d_edges + value
        pos_edges = np.searchsorted(het_map[i+1:], edges_for_i)
        pair_counts += np.diff(pos_edges)
        #
        if np.sum(np.diff(pos_edges)) != n_het - i - 1:
            exp = n_het - i - 1
            print(f"{i}, exp {exp} got {np.sum(np.diff(pos_edges))}")
        #
        if i % 1e3 == 0:
            print(f"{i} bp scanned, {np.sum(pair_counts)} pairs binned")
    print(f"{i} bp scanned, {np.sum(pair_counts)} pairs binned")
    expected_n_pairs = int(n_het * (n_het - 1) / 2)
    n_pairs = int(np.sum(pair_counts))
    diff = int(n_pairs - expected_n_pairs)
    print(f"{n_pairs} recorded out of {expected_n_pairs}, difference {diff}")
    return pair_counts


def setup_bootstrap(dir, pair_counts, sample="tsk_0"):
    files = [dir + file for file in os.listdir(dir)]
    bed_path = [file for file in files if ".bed" in file][0]
    pi_2s = []
    for file in files:
        if ".vcf.gz" in file:
            vec = GenotypeVector.read_abbrev_vcf(file, bed_path, sample)
            hets = get_het_pair_distribution(vec, maskedmap, r_edges)
            pi_2s.append(hets/pair_counts)
    return pi_2s


def bootstrap(pi_2s, B, x):
    pi_2s = np.array(pi_2s)
    n_samples = len(pi_2s)
    means = np.zeros((B, len(pi_2s[0])))
    for i in np.arange(B):
        samples = np.random.randint(0, n_samples, x)
        means[i, :] = np.mean(pi_2s[samples], axis=0)
    return means


def load_simulated_vcfs(dir, r_edges):
    hets = np.zeros(len(r_edges) - 1, dtype=np.int64)
    files = os.listdir(dir)
    for file in files:
        vec = GenotypeVector.read_abbrev_vcf(
            dir + '/' + file,
            "c:/archaic/data/chromosomes/merged_masks/chr22/chr22_merge.bed",
            "tsk_0")
        hets += get_het_pair_distribution(vec, maskedmap, r_edges)
    return hets
