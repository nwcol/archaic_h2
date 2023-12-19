import sys

sys.path.insert(0, "c:/archaic/src")

import statistics

from archaic import vcf_util


def main(file_name, sample):
    samples = vcf_util.read_sample_ids(file_name)
    if sample.encode() not in samples:
        raise ValueError("Invalid sample specified")
    alt = vcf_util.read_sample(file_name, sample)
    n = 2
    diversity = statistics.compute_diversity(alt, n)
    return diversity


file_name = sys.argv[1]
sample = sys.argv[2]
print(main(file_name, sample))
