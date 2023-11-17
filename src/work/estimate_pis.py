# estimate all diversities and divergences given the samples in a .vcf file

import numpy as np

import sys

import vcf_util

import statistics


def main(file_name):
    samples = [x.decode() for x in vcf_util.parse_samples(file_name)]
    n_samples = len(samples)
    n = 2
    idxs = []
    for i in np.arange(n_samples):
        for j in np.arange(0, i):
            idxs.append((i, j))
    sample_pairs = [(samples[i], samples[j]) for i, j in idxs]
    alt_vectors = dict()
    for sample in samples:
        alt_vectors[sample] = vcf_util.read_genotypes(file_name, sample)
        pi = statistics.compute_diversity(alt_vectors[sample], n)
        print(f"pi_{sample} = {np.round(pi, 6)} \n")
    for sample_0, sample_1 in sample_pairs:
        alt_0 = alt_vectors[sample_0]
        alt_1 = alt_vectors[sample_1]
        divergence = statistics.compute_divergence(alt_0, alt_1, n, n)
        print(f"pi_({sample_0}, {sample_1}) = {np.round(divergence, 6)} \n")

file_name = sys.argv[1]
main(file_name)
