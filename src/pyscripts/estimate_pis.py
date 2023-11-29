# estimate all diversities and divergences given the samples in a .vcf file

import numpy as np

import sys

import vcf_util

import statistics


def main(file_name):
    samples = [x.decode() for x in vcf_util.parse_samples(file_name)]
    samples.sort()
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
        pi_x = statistics.compute_diversity(alt_vectors[sample], n)
        out = dict(statistic="pi_x", sample=(sample,), value=pi_x)
        print(str(out))
    for sample_0, sample_1 in sample_pairs:
        alt_0 = alt_vectors[sample_0]
        alt_1 = alt_vectors[sample_1]
        pi_xy = statistics.compute_divergence(alt_0, alt_1, n, n)
        out = dict(statistic="pi_xy", sample=(sample_0, sample_1), value=pi_xy)
        print(str(out))


name = sys.argv[1]
main(name)
