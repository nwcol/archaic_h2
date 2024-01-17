
# Merge with coalescent.py at some point ~

import demes

import demesdraw

from IPython.display import display

import matplotlib.pyplot as plt

import matplotlib

import msprime

import numpy as np

import time

import os

import archaic.bed_util as bed_util

import archaic.vcf_util as vcf_util

import archaic.map_util as map_util


if __name__ == "__main__":
    plt.rcParams['figure.dpi'] = 100
    matplotlib.use('Qt5Agg')


map_path = "c:/archaic/data/chromosomes/maps/chr22/genetic_map_GRCh37_chr22.txt"


def get_ts(map_path=map_path, N_e=1e4, L=52e6, u=1.5e-8):

    rate_map = msprime.RateMap.read_hapmap(map_path, sequence_length=L)
    ts = msprime.sim_ancestry(samples=1,
                              ploidy=2,
                              population_size=N_e,
                              sequence_length=L,
                              discrete_genome=True,
                              recombination_rate=rate_map
                              )
    ts = msprime.sim_mutations(ts, rate=u)
    return ts


def write_ts(ts, path, chr=22):
    file = open(path, 'w')
    ts.write_vcf(file, contig_id=22)
    file.close()
    return 0





















