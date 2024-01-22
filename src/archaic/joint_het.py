
#

import numpy as np

import sys

import matplotlib

import matplotlib.pyplot as plt

import time

sys.path.append("/home/nick/Projects/archaic/src")

import archaic.vcf_samples as vcf_samples

import archaic.map_util as map_util


if __name__ == "__main__":
    plt.rcParams['figure.dpi'] = 100
    matplotlib.use('Qt5Agg')


edges = np.array([0,
                  1e-7, 2e-7, 5e-7,
                  1e-6, 2e-6, 5e-6,
                  1e-5, 2e-5, 5e-5,
                  1e-4, 2e-4, 5e-4,
                  1e-3, 2e-3, 5e-3,
                  1e-2, 2e-2, 5e-2,
                  1e-1, 2e-1, 5e-1], dtype=np.float64)

r = edges[1:]



def naive(samples, sample_id_x, sample_id_y, r_edges=edges):

    time_0 = time.time()

    alt_idx = samples.alt_index
    alt_map = samples.alt_map_values

    alts_x = samples.samples[sample_id_x]
    alts_y = samples.samples[sample_id_y]

    n_alts = samples.n_variants

    n_bins = len(r_edges) - 1
    joint_het = np.zeros(n_bins)


    for i in np.arange(n_alts):

        if alts_x[i] != 0 or alts_y[i] != 0:

            for j in np.arange(i + 1, n_alts):

                if alts_x[j] != 0 or alts_y[i] != 0:

                    probs_x = get_hap_probs(alts_x[i], alts_x[j])
                    probs_y = get_hap_probs(alts_y[i], alts_y[j])
                    het = np.sum(probs_x * np.flip(probs_y))
                    r_dist = map_util.d_to_r(alt_map[j] - alt_map[i])
                    selector = np.searchsorted(r_edges, r_dist) - 1
                    if selector == 21:
                        print(alt_map[i], alt_map[j], r_dist)
                        break
                    joint_het[selector] += het

        if i % 100 == 0 and i > 0:
            print(i, np.round(time.time() - time_0, 2))

        if i > 100:
            break

    return joint_het


def get_hap_probs(alts_i, alts_j):

    A = alts_i / 2
    a = 1 - A
    B = alts_j / 2
    b = 1 - B
    return np.array([A * B, A * b, a * B, a * b])


def simple(samples, sample_id_x, sample_id_y, r_edges=edges):

    time_0 = time.time()

    alt_map = samples.alt_map_values

    alts_x = samples.samples[sample_id_x]
    alts_y = samples.samples[sample_id_y]

    freq_x = (alts_x / 2).astype(np.float64)
    freq_y = (alts_y / 2).astype(np.float64)

    n_alts = samples.n_variants

    n_bins = len(r_edges) - 1
    joint_het = np.zeros(n_bins)

    for i in np.arange(n_alts):

        if alts_x[i] != 0 or alts_y[i] != 0:

            pr_x = vec_probs(freq_x[i], freq_x[i + 1:])
            pr_y = vec_probs(freq_y[i], freq_y[i + 1:])

            pr_y = np.flip(pr_y, axis=0)

            hets = np.sum(pr_x * pr_y, axis=0)

            d_dist = alt_map[i + 1:] - alt_map[i]
            r_dist = map_util.d_to_r(d_dist)
            bin_assigns = np.searchsorted(r_edges, r_dist) - 1

            for b in np.arange(n_bins):
                joint_het[b] += np.sum(hets[bin_assigns == b])

        if i % 1000 == 0 and i > 0:
            print(i, np.round(time.time() - time_0, 2))

    return joint_het


def vec_probs(A, B_vec):

    a = 1 - A
    b_vec = 1 - B_vec
    probs = np.array([A * B_vec, A * b_vec, a * B_vec, a * b_vec],
                     dtype=np.float64)
    return probs










































def main(samples, sample_id_x, sample_id_y, r_edges=edges):
    """
    loop over variant sites

    at each variant site, manually compute joint het with other variant sites

    then add joint het from invariant 0/0 sites iff the variant site is 1/1
    else this can be skipped
    """

    alt_index = samples.alt_index
    alts_x = samples.samples[sample_id_x]
    alts_y = samples.samples[sample_id_y]



    alt_map = samples.alt_map_values



    n_positions = samples.n_positions
    n_bins = len(r_edges) - 1

    het = np.zeros(n_bins)
    n = 0

    for i, alt_idx in enumerate(alt_index):

        hap_probs_x = get_hap_probs_x(alts_x[i], alts_x[i + 1:])
        hap_probs_y = get_hap_probs_y(alts_y[i], alts_y[i + 1:])


        bins = bin_alts(i, alt_idx, alt_map, r_edges=r_edges)
        hets = get_joint_hets(hap_probs_x, hap_probs_y)

        for j in range(n_bins):

            het[j] += np.sum(hets[bins == j])


        # n_het += get_joint_het(hap_probs_x, hap_probs_y)

        n_above = n_positions - alt_idx
        n += n_above

        verbose(i)

    return n, het


def get_map_bins(map_dists, r_edges):

    r_dists = map_util.d_to_r(map_dists)
    bins = np.searchsorted(r_edges, r_dists)
    return bins


def bin_alts(i, alt_idx, alt_map, r_edges):

    map_dists = alt_map[i + 1:] - alt_idx
    return get_map_bins(map_dists, r_edges)


def get_joint_hets(hap_probs_x, hap_probs_y):

    hets = np.sum(hap_probs_x * hap_probs_y, axis=0)
    return hets


def verbose(i):
    if i % 1_000 == 0 and i > 0:
        print(i)
































"""
idea: binning pairs based on distance handled by one function, which then 
hands the pair array to one which computes het.

this means you dont have to worry about binning. ignore it for now

challenges: 
- each site may have several nonzero haplotype probabilities




haplotypes in one diploid site: (A derived)
A|A, A|a, a|A, a|a

or 
B|B, B|b, b|B, b|b 


haplotypes in two diploid sites:
A|A, A|a, a|A, a|a
B|B, B|B, B|B, B|B 

A|A, A|a, a|A, a|a
B|b, B|b, B|b, B|b

A|A, A|a, a|A, a|a
b|B, b|B, b|B, b|B

A|A, A|a, a|A, a|a
b|b, b|b, b|b, b|b


haploid samples from a diploid at two sites:
AB, Ab, aB, ab


Combinations of samples from two diploids:
AB, Ab, aB, ab
AB, AB, AB, AB

AB, Ab, aB, ab
Ab, Ab, Ab, Ab

AB, Ab, aB, ab
aB, aB, aB, aB

AB, Ab, aB, ab
ab, ab, ab, ab


The entries along the rising diagonal AB-ab, Ab-aB, aB-Ab, ab-AB contribute to 
joint heterozygosity. So heterozygosity for a site pair for samples x, y equals

P_x(AB)P_y(ab) + P_x(Ab)P_y(aB) + P_x(aB)P_y(Ab) + P_x(ab)P_y(AB)





clearly for all invariant sites, the vector of sample hap probs
<P_x(AB), P_x(Ab), P_x(aB), P_x(ab)> = <0, 0, 0, 1>
so I do not need to compute it

"""


# goal: set things up in a highly atomized way to understand what you're doing





def get_hap_probs_x(alt_count_0, alt_counts):
    """
    a: reference allele freq at site 0
    A: alternate allele freq at site 0
    b: ref allele freq at site 1
    B: alt allele freq at site 1


    :param alt_count_0:
    :param alt_counts:
    :return:
    """
    A = alt_count_0 / 2
    a = 1 - A
    B = alt_counts / 2
    b = 1 - B
    hap_probs = np.array([A * B, A * b, a * B, a * b], dtype=np.float64)
    return hap_probs


def get_hap_probs_y(alt_count_0, alt_counts):
    """
    a: reference allele freq at site 0
    A: alternate allele freq at site 0
    b: ref allele freq at site 1
    B: alt allele freq at site 1


    :param alt_count_0:
    :param alt_counts:
    :return:
    """
    A = alt_count_0 / 2
    a = 1 - A
    B = alt_counts / 2
    b = 1 - B
    hap_probs = np.array([a * b, a * B, A * b, A * B], dtype=np.float64)
    return hap_probs


def get_joint_het(hap_probs_x, hap_probs_y):
    """


    :return:
    """
    n_het = np.sum(hap_probs_x * hap_probs_y)
    return n_het








def joint_het(alts_0, alts_1):
    """
    input

    np.array([[sample x site 0 alts, sample x site 1 alts],
              [sample y site 0 alts, sample y site 1 alts]])


    :param matrix:
    :return:
    """
    x_probs = hap_probs(alts_0 / 2)
    y_probs = hap_probs(alts_1 / 2)
    return np.sum(x_probs * np.flip(y_probs))


def hap_probs(alt_freqs):
    A = alt_freqs[0]
    a = 1 - A
    B = alt_freqs[1]
    b = 1 - B
    return np.array([a * b, a * B, A * b, A * B], dtype=np.float64)


def test(alts_0, alts_1):

    n = len(alts_0)

    matrix = np.zeros((n, n), dtype=np.float64)

    for i in np.arange(n):

        for j in np.arange(i + 1, n):

            matrix[i, j] = joint_het(np.array([alts_0[i], alts_0[j]]),
                                     np.array([alts_1[i], alts_1[j]]))

    return np.round(matrix, 2)


def enumerate_pairs(items):
    n = len(items)
    pairs = []
    for i in np.arange(n):
        for j in np.arange(i + 1, n):
            pairs.append((items[i], items[j]))
    return pairs












chr22_samples = vcf_samples.UnphasedSamples.dir(
    "/home/nick/Projects/archaic/data/chromosomes/merged/chr22/")

