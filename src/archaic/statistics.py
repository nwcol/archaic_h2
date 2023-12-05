
# Functions for computing statistics from vectors of alternate allele counts

import matplotlib.pyplot as plt

import matplotlib

import numpy as np

import time

from bed_util import Bed

import vcf_util

from map_util import Map, MaskedMap

import map_util


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


def compute_pi_2(genotype_vector, masked_map, r_bins, max_i=None):
    n_bins = len(r_bins)
    d_bins = map_util.r_bins_to_d_bins(r_bins)
    sum_pairs = np.zeros(n_bins, dtype=np.int64)
    sum_joint_het = np.zeros(n_bins, dtype=np.int64)
    if not max_i:
        max_i = genotype_vector.length
    for i in np.arange(max_i):
        pos_bins = masked_map.approximate_r_bins(i, d_bins)
        counts = pos_bins[:, 1] - pos_bins[:, 0]
        sum_pairs += counts
        if genotype_vector.het_indicator[i] == 1:
            sum_joint_het += compute_n_joint_hets(genotype_vector, i, pos_bins)
        if i % 1e6 == 0:
            print(f"{i} bp complete, {np.sum(sum_pairs)} pairs computed")
    print(f"{i} bp complete, {np.sum(sum_pairs)} pairs computed")
    pi_2 = np.zeros(n_bins)
    non_zero = np.where(sum_pairs > 0)[0]
    pi_2[non_zero] = sum_joint_het[non_zero] / sum_pairs[non_zero]
    return sum_joint_het, sum_pairs


def linear_approx_pi_2(genotype_vector, r_bins, max_i=None):
    n_bins = len(r_bins)
    base_bins = r_bins * 1e8  # linear approximation
    base_bins = base_bins.astype(dtype=np.int64)
    sum_pairs = np.zeros(n_bins, dtype=np.int64)
    sum_joint_het = np.zeros(n_bins, dtype=np.int64)
    if not max_i:
        max_i = genotype_vector.length
    for i in np.arange(max_i):
        binz = base_bins + genotype_vector.positions[i]
        pos_bins = np.searchsorted(genotype_vector.positions, + binz)
        counts = pos_bins[:, 1] - pos_bins[:, 0]
        sum_pairs += counts
        if genotype_vector.het_indicator[i] == 1:
            sum_joint_het += compute_n_joint_hets(genotype_vector, i, pos_bins)
        if i % 1e6 == 0:
            print(f"{i} bp complete, {np.sum(sum_pairs)} pairs computed")
    print(f"{i} bp complete, {np.sum(sum_pairs)} pairs computed")
    pi_2 = np.zeros(n_bins, dtype=np.float64)
    non_zero = np.where(sum_pairs > 0)[0]
    pi_2[non_zero] = sum_joint_het[non_zero] / sum_pairs[non_zero]
    return sum_joint_het, sum_pairs


def compute_n_joint_hets(genotype_vector, i, pos_bins):
    if genotype_vector.het_indicator[i] != 1:
        het_sum = np.zeros(len(pos_bins), dtype=np.int64)
    else:
        het_sum = np.zeros(len(pos_bins), dtype=np.int64)
        counts = pos_bins[:, 1] - pos_bins[:, 0]
        for i, pos_bin in enumerate(pos_bins):
            if counts[i] > 0:
                b0, b1 = pos_bin
                het_sum[i] = np.sum(genotype_vector.het_indicator[b0:b1])
    return het_sum


def joint_het_distribution(genotype_vector, masked_map, r_bins):
    """
    Return a vector of heterozygote pair counts per r bin

    :param genotype_vector:
    :param maskedmap:
    :param i:
    :param r_bins:
    :return:
    """
    n_bins = len(r_bins)
    het_pair_count = np.zeros(n_bins, dtype=np.int64)
    het_index = genotype_vector.het_index
    map_values = masked_map.map_values
    for i in het_index:
        right_hets = het_index[het_index > i]
        d_distance = map_values[right_hets] - map_values[i]
        distances = d_to_r(d_distance)
        for k, (b0, b1) in enumerate(r_bins):
            count = np.sum((distances >= b0) & (distances < b1))
            het_pair_count[k] += count
    return het_pair_count


def d_to_r(d_distance):
    """
    Use Haldane's map function to convert cM to r

    :param d_distance:
    :return:
    """
    r_distance = 0.5 * (1 - np.exp(-0.02 * d_distance))
    return r_distance


def r_to_d(r_distance):
    """
    Use Haldane's map function to convert r to cM

    :param r_distance:
    :return:
    """
    d_distance = np.log(1 - 2 * r_distance) / -0.02
    return d_distance


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


r_bins_0 = np.array([[0, 1e-7],
                     [1e-7, 1e-6],
                     [1e-6, 1e-5],
                     [1e-5, 1e-4],
                     [1e-4, 1e-3],
                     [1e-3, 1e-2],
                     [1e-2, 1e-1],
                     [1e-1, 5e-1]], dtype=np.float32)

r_bins_1 = np.array([[0, 1e-8], [1e-8, 2e-8], [2e-8, 5e-8], [5e-8, 1e-7],
                     [1e-7, 2e-7], [2e-7, 5e-7], [5e-7, 1e-6],
                     [1e-6, 2e-6], [2e-6, 5e-6], [5e-6, 1e-5],
                     [1e-5, 2e-5], [2e-5, 5e-5], [5e-5, 1e-4],
                     [1e-4, 2e-4], [2e-4, 5e-4], [5e-4, 1e-3],
                     [1e-3, 2e-3], [2e-3, 5e-3], [5e-3, 1e-2],
                     [1e-2, 2e-2], [2e-2, 5e-2], [5e-2, 1e-1],
                     [1e-1, 2e-1], [2e-1, 5e-1]], dtype=np.float32)

r_bins_2 = np.array([[0, 1e-7], [1e-7, 2e-7], [2e-7, 5e-7], [5e-7, 1e-6],
                     [1e-6, 2e-6], [2e-6, 5e-6], [5e-6, 1e-5],
                     [1e-5, 2e-5], [2e-5, 5e-5], [5e-5, 1e-4],
                     [1e-4, 2e-4], [2e-4, 5e-4], [5e-4, 1e-3],
                     [1e-3, 2e-3], [2e-3, 5e-3], [5e-3, 1e-2],
                     [1e-2, 2e-2], [2e-2, 5e-2], [5e-2, 1e-1],
                     [1e-1, 2e-1], [2e-1, 5e-1]], dtype=np.float32)


_map = Map.load_txt(
    "c:/archaic/data/chromosomes/maps/chr22/genetic_map_GRCh37_chr22.txt")
_bed = Bed.load_bed(
    "c:/archaic/data/chromosomes/merged_masks/chr22/chr22_merge.bed")
maskedmap = MaskedMap.from_class(map, _bed)
genotypevector = GenotypeVector.read_vcf(
    "c:/archaic/data/chromosomes/merged/chr22/complete_chr22.vcf.gz",
    "SS6004475")
