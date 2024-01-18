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


def main():

    return 0









def site_pair_joint_het(hap_p_x, hap_p_y):
    """
    Return the expected joint heterozygosity for a single site pair

    :return:
    """
    return np.sum(hap_p_x * np.flip(hap_p_y))














chr22_samples = vcf_samples.UnphasedSamples.dir(
    "/home/nick/Projects/archaic/data/chromosomes/merged/chr22/")

