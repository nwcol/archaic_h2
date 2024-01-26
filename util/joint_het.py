
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
    alt_map = samples.alt_map_values
    freq_x = (samples.samples[sample_id_x] / 2).astype(np.float64)
    freq_y = (samples.samples[sample_id_y] / 2).astype(np.float64)
    n_alts = samples.n_variants
    n_bins = len(r_edges) - 1
    joint_het = np.zeros(n_bins)

    k = 0

    for i in np.arange(n_alts):

        if freq_x[i] != 0 or freq_y[i] != 0:

            bin_idx = assign_bins(i, alt_map, r_edges)

            for j in np.arange(i + 1, n_alts):

                if freq_x[j] != 0 or freq_y[j] != 0:

                    probs_x = get_haplotype_probs(freq_x[i], freq_x[j])
                    probs_y = get_haplotype_probs(freq_y[i], freq_y[j])
                    het = np.sum(probs_x * np.flip(probs_y))
                    # r_dist = map_util.d_to_r(alt_map[j] - alt_map[i])
                    # selector = np.searchsorted(r_edges, r_dist) - 1
                    # joint_het[selector] += het

                    joint_het[bin_idx[j - i - 1]] += het

                    k += 1

        if i % 100 == 0 and i > 0:
            print(i, np.round(time.time() - time_0, 2))

        if i > 500:
            break

    print(k)
    return joint_het


def get_haplotype_probs(A, B):

    a = 1 - A
    b = 1 - B
    return np.array([A * B, A * b, a * B, a * b])


def two_pop_joint_het(samples, sample_id_x, sample_id_y, r_edges=edges):
    """
    Compute cross-population joint heterozygosity in a Samples instance for
    samples named by the sample_ids.

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
            print(i, np.round(time.time() - time_0, 2))

    return joint_het





def joint_het(alts_0, alts_1):
    """
    input

    np.array([[sample x site 0 alts, sample x site 1 alts],
              [sample y site 0 alts, sample y site 1 alts]])


    :param alts_0:
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





samples = vcf_samples.UnphasedSamples.dir(
    "/home/nick/Projects/archaic/data/chromosomes/merged/chr22/")


