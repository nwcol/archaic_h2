
# Functions for computing statistics from vectors of alternate allele counts

import matplotlib.pyplot as plt

import matplotlib

import numpy as np

import time

import os

import sys

from src.archaic import vcf_util

from src.archaic.bed_util import Bed

from map_util import Map, MaskedMap

import map_util


if __name__ == "__main__":
    plt.rcParams['figure.dpi'] = 100
    matplotlib.use('Qt5Agg')


# functions that operate on a GenotypeVector



r_edges = np.array([0,
                    1e-7, 2e-7, 5e-7,
                    1e-6, 2e-6, 5e-6,
                    1e-5, 2e-5, 5e-5,
                    1e-4, 2e-4, 5e-4,
                    1e-3, 2e-3, 5e-3,
                    1e-2, 2e-2, 5e-2,
                    1e-1, 2e-1, 5e-1], dtype=np.float32)

r = r_edges[1:]


bed = Bed.load_bed("c:/archaic/data/chromosomes/merged/chr22_og/chr22_merge.bed")
map = Map.load_txt("c:/archaic/data/chromosomes/maps/GRCh37/genetic_map_GRCh37_chr22.txt")

maskedmap = MaskedMap.from_class(map, bed)

#og_vector = GenotypeVector.read_vcf("c:/archaic/data/chromosomes/merged/chr22_og/complete_chr22.vcf.gz", "Denisova")
#new_vector = GenotypeVector.read_vcf("c:/archaic/TEST/chr22_merged.vcf.gz", "Denisova")
