
# Functions for computing statistics from vectors of alternate allele counts

import matplotlib.pyplot as plt

import matplotlib

import numpy as np

from bed_util import Bed

import vcf_util

from map_util import Map, MaskedMap


if __name__ == "__main__":
    plt.rcParams['figure.dpi'] = 100
    matplotlib.use('Qt5Agg')


class GenotypeVector:
    """
    Class to keep track of two paired vectors holding genotypes and the
    genotype positions
    """

    def __init__(self, positions, genotypes):
        """
        Position and code vectors must have matching lengths

        :param positions: vector of positions, 0 indexed
        :param genotypes: vector of genotype codes
        """
        if len(positions) != len(genotypes):
            raise ValueError("vector lengths must match")
        self.positions = positions
        self.genotypes = genotypes
        self.length = len(positions)
        self.het_sites = positions[genotypes == 1]
        self.het_index = np.nonzero(self.genotypes == 1)[0]
        self.het_indicator = np.zeros(self.length, dtype=np.uint8)
        self.het_indicator[self.genotypes == 1] = 1
        self.position_index = np.arange(self.length)

    @classmethod
    def read_vcf(cls, vcf_path, sample, bed_path=None):
        if not bed_path:
            bed = Bed.from_vcf(vcf_path)
        else:
            bed = Bed.load_bed(bed_path)
        positions = bed.get_positions_1()
        n_positions = bed.n_positions
        genotypes = vcf_util.read_genotypes(vcf_path, sample, n_positions)
        return cls(positions, genotypes)

    @property
    def n_het(self):
        """
        Return the number of heterozygous sites

        :return:
        """
        return np.sum(self.genotypes == 1)

    def get_window_index(self, window):
        """
        Turn a window (slice) into a vector of indices

        :param window:
        :return:
        """
        index = np.where((self.positions > window[0])
                         & (self.positions < window[1]))[0]
        return index

    def window_n_het(self, window):
        """
        Return the number of heterozygous sites in a given window (slice)

        :param window: slice
        :return:
        """
        index = self.get_window_index(window)
        return np.sum(self.genotypes[index] == 1)

    def window_length(self, window):
        """
        Return the number of positions represented in a given window

        :param window: slice
        :return:
        """
        return len(self.get_window_index(window))

    def compute_pi_x(self):
        """
        Compute an estimator for diversity throughout the whole vector

        :return:
        """
        ref = 2 - self.genotypes
        tot = self.genotypes * ref
        pi_x = np.sum(tot) / self.length
        return pi_x

    def compute_H(self):
        """
        Should exactly equal pi_x

        :return:
        """
        h = self.n_het
        H = h / self.length
        return H

    def compute_joint_H(self, pos, window):
        """
        compute joint heterozygosity for a site in a range

        :return:
        """
        window_het = self.window_n_het(window)
        window_length = self.window_length(window)
        joint_H = window_het / window_length
        return joint_H


# functions that operate on a GenotypeVector

def manual_joint_het(genotype_vector, masked_map, i, r_0, r_1):
    """
    Testing!!

    :param genotype_vector:
    :param masked_map:
    :param i:
    :param r_0:
    :param r_1:
    :return:
    """
    if genotype_vector.het_indicator[i] != 1:
        pi_2 = 0
    else:
        sites = masked_map.sites_in_r(i, r_0, r_1)
        n_sites = len(sites)
        n_hets = np.sum(genotype_vector.het_indicator[sites])
        if n_sites > 0:
            pi_2 = n_hets / n_sites
        else:
            pi_2 = 0
    return pi_2


def test(genotype_vector, masked_map, max_i):
    n_r_bins = len(r_bins)
    x = genotype_vector.position_index  # vector of position indices
    for i in np.arange(max_i + 1):
        r_vec = masked_map.compute_r(i)
        for k, r_0, r_1 in enumerate(r_bins):
            sites = np.nonzero((r_vec > r_0) & (r_vec <= r_1) & (x > i))
            hets = genotype_vector.het_indicator[sites]


def count_pairs(masked_map, genotype_vector, min_i, max_i):
    n_pairs_per_site = np.zeros((max_i, len(r_bins)))
    x = genotype_vector.position_index
    for i in np.arange(min_i, max_i):
        r_vec = masked_map.compute_r(i)
        for k, r_bin in enumerate(r_bins):
            r_0, r_1 = r_bin
            sites = np.nonzero((r_vec > r_0) & (r_vec <= r_1) & (x > i))[0]
            n_pairs_per_site[i, k] = len(sites)
    return n_pairs_per_site


# functions that take alt indicator vectors or other statistics as arguments

def compute_diversity(alt_x, n_x):
    """
    Compute an estimator for diversity from a vector of alternate allele counts

    :param alt_x:
    :param n_x:
    :return:
    """
    length = len(alt_x)
    ref_x = n_x - alt_x
    tot = alt_x * ref_x
    coeff = 2 / (length * n_x * (n_x - 1))
    diversity = coeff * np.sum(tot)
    return diversity


def compute_divergence(alt_x, alt_y, n_x, n_y):
    """
    Compute an estimator for divergence from two vectors of alternate allele
    counts.

    :param alt_x:
    :param alt_y:
    :param n_x:
    :param n_y:
    :return:
    """
    length = len(alt_x)
    if len(alt_y) != length:
        raise ValueError("Alt vector lengths do not match")
    ref_x = n_x - alt_x
    ref_y = n_y - alt_y
    tot = (ref_x * alt_y) + (alt_x * ref_y)
    coeff = 1 / (length * n_x * n_y)
    divergence = coeff * np.sum(tot)
    return divergence


def compute_F_2(pi_1, pi_2, pi_12):
    """
    Compute an estimator for the F2 statistic

    :return:
    """
    F_2 = pi_12 - (pi_1 + pi_2) / 2
    return F_2


def compute_F_3(pi_x, pi_1, pi_2, pi_x1, pi_x2, pi_12):
    """
    Compute an estimator for the F_3 statistic

    :param pi_x:
    :param pi_1:
    :param pi_2:
    :param pi_x1:
    :param pi_x2:
    :param pi_12:
    :return:
    """
    F_2_x1 = compute_F_2(pi_x1, pi_x, pi_1)
    F_2_x2 = compute_F_2(pi_x2, pi_x, pi_2)
    F_2_12 = compute_F_2(pi_12, pi_1, pi_2)
    F_3 = 0.5 * F_2_x1 * F_2_x2 * F_2_12
    return F_3


def compute_F_4():
    pass


def compute_F_st():
    pass


_r_bins = [(0, 1e-7), (1e-7, 2e-7), (2e-7, 3e-7), (3e-7, 5e-7), (5e-7, 1e-6),
          (1e-6, 2e-6), (2e-6, 3e-6), (3e-6, 5e-6), (5e-6, 1e-5),
          (1e-5, 2e-5), (2e-5, 3e-5), (3e-5, 5e-5), (5e-5, 1e-4),
          (1e-4, 2e-4), (2e-4, 3e-4), (3e-4, 5e-4), (5e-4, 1e-4),
          (1e-3, 2e-3), (2e-3, 3e-3), (3e-3, 5e-3), (5e-3, 1e-3),
          (1e-2, 2e-2), (2e-2, 3e-2), (3e-2, 5e-2), (5e-2, 1e-2),
          (1e-1, 2e-1), (2e-1, 3e-1), (3e-1, 5e-1)]

r_bins = [(0, 1e-7), (2e-7, 5e-7), (2e-7, 5e-7),
          (1e-6, 2e-6), (2e-6, 5e-6), (5e-6, 1e-5),
          (1e-5, 2e-5), (2e-5, 5e-5), (5e-5, 1e-4),
          (1e-4, 2e-4), (2e-4, 5e-4), (5e-4, 1e-4),
          (1e-3, 2e-3), (2e-3, 5e-3), (5e-3, 1e-3),
          (1e-2, 2e-2), (2e-2, 5e-2), (5e-2, 1e-2),
          (1e-1, 2e-1), (2e-1, 5e-1)]

map = Map.load_txt(
    "c:/archaic/data/chromosomes/maps/chr22/genetic_map_GRCh37_chr22.txt")
bed = Bed.load_bed(
    "c:/archaic/data/chromosomes/merged_masks/chr22/chr22_merge.bed")
maskedmap = MaskedMap.from_class(map, bed)
genotype_vector = GenotypeVector.read_vcf(
    "c:/archaic/data/chromosomes/merged/chr22/complete_chr22.vcf.gz",
    "Denisova")
