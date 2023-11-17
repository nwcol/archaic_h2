import sys

import statistics

import vcf_util


def main(file_name, sample_0, sample_1):
    samples = vcf_util.parse_samples(file_name)
    if sample_0.encode() not in samples:
        raise ValueError("Invalid sample_0 specified")
    if sample_1.encode() not in samples:
        raise ValueError("Invalid sample_1 specified")
    alt_0 = vcf_util.read_genotypes(file_name, sample_0)
    alt_1 = vcf_util.read_genotypes(file_name, sample_1)
    n = 2
    divergence = statistics.compute_divergence(alt_0, alt_1, n, n)
    return divergence


file_name = sys.argv[1]
sample_0 = sys.argv[2]
sample_1 = sys.argv[3]
print(main(file_name, sample_0, sample_1))
