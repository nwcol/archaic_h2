
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


def get_joint_het_dist(genotype_vector, masked_map, r_bins):
    """
    Older function

    :param genotype_vector:
    :param masked_map:
    :param r_bins: bins, given in recombination frequencies r
    :return:
    """
    n_bins = len(r_bins)
    n_het = genotype_vector.n_het
    het_pair_counts = np.zeros(n_bins, dtype=np.int64)
    het_map = masked_map.values[genotype_vector.het_index]
    for i in np.arange(n_het):
        right_hets = het_map[i+1:]
        focus = het_map[i]
        d_distance = right_hets - focus
        r_distances = map_util.d_to_r(d_distance)
        for k, (b0, b1) in enumerate(r_bins):
            count = np.sum((r_distances >= b0) & (r_distances < b1))
            het_pair_counts[k] += count
    expected_n_pairs = n_het * (n_het - 1) / 2
    n_pairs = int(np.sum(het_pair_counts))
    diff = n_pairs - expected_n_pairs
    print(f"{n_pairs} recorded out of {expected_n_pairs}, difference {diff}")
    return het_pair_counts


def linear_pair_dist(genotype_vector, r_bins):
    """
    Use an unrealistic linear model of r to compute bin pair counts.

    For prototyping

    :param genotype_vector:
    :param r_bins:
    :return:
    """
    n_bins = len(r_bins)
    n_pos = genotype_vector.length
    pair_counts = np.zeros(n_bins, dtype=np.int64)
    pos_bins = (r_bins * 1e8).astype(np.int64)
    # convert bins in r to bins in bp
    i = 0
    for i in np.arange(n_pos):
        pos = genotype_vector.positions[i]
        focal_bins = pos_bins + pos
        edges = np.searchsorted(genotype_vector.positions, focal_bins)
        edges[0, 0] = i + 1
        pair_counts += edges[:, 1] - edges[:, 0]
        #
        if i % 1e6 == 0:
            print(f"{i} bp scanned, {np.sum(pair_counts)} pairs binned")
    print(f"{i + 1} bp scanned, {np.sum(pair_counts)} pairs binned")
    expected_n_pairs = int(n_pos * (n_pos - 1) / 2)
    n_pairs = int(np.sum(pair_counts))
    diff = int(n_pairs - expected_n_pairs)
    print(f"{n_pairs} recorded out of {expected_n_pairs}, difference {diff}")
    return pair_counts


def linear_het_pair_dist(genotype_vector, r_bins):
    """
    Use an unrealistic linear model of r to get the number of site pairs per
    bin for an array of bins in r

    :param genotype_vector:
    :param masked_map:
    :param r_bins:
    :param max_i:
    :return:
    """
    n_bins = len(r_bins)
    n_het = genotype_vector.n_het
    pair_counts = np.zeros(n_bins, dtype=np.int64)
    pos_bins = (r_bins * 1e8).astype(np.int64)
    het_pos = genotype_vector.het_sites
    i = 0
    for i in np.arange(n_het):
        pos = het_pos[i]
        focal_bins = pos_bins + pos
        edges = np.searchsorted(het_pos, focal_bins)
        edges[0, 0] = i + 1
        pair_counts += edges[:, 1] - edges[:, 0]
        #
        if i % 1e3 == 0:
            print(f"{i} bp scanned, {np.sum(pair_counts)} pairs binned")
    print(f"{i + 1} bp scanned, {np.sum(pair_counts)} pairs binned")
    expected_n_pairs = int(n_het * (n_het - 1) / 2)
    n_pairs = int(np.sum(pair_counts))
    diff = int(n_pairs - expected_n_pairs)
    print(f"{n_pairs} recorded out of {expected_n_pairs}, difference {diff}")
    return pair_counts


def get_pair_distribution(genotype_vector, masked_map, r_bins):
    """
    Get the number of site pairs per bin for an array of bins in r

    :param genotype_vector:
    :param masked_map:
    :param r_bins:
    :param max_i:
    :return:
    """
    n_bins = len(r_bins)
    n_pos = genotype_vector.length
    pair_counts = np.zeros(n_bins, dtype=np.int64)
    d_bins = map_util.r_to_d(r_bins)  # convert bins in r to bins in d
    i = 0
    for i in np.arange(n_pos):
        focus = masked_map.values[i]
        focal_bins = d_bins + focus
        pos_bins = np.searchsorted(masked_map.map_values, focal_bins)
        pos_bins[0, 0] = i + 1
        pair_counts += pos_bins[:, 1] - pos_bins[:, 0]
        #
        if i % 1e6 == 0:
            print(f"{i} bp scanned, {np.sum(pair_counts)} pairs binned")
    print(f"{i} bp scanned, {np.sum(pair_counts)} pairs binned")
    expected_n_pairs = int(n_pos * (n_pos - 1) / 2)
    n_pairs = int(np.sum(pair_counts))
    diff = int(n_pairs - expected_n_pairs)
    print(f"{n_pairs} recorded out of {expected_n_pairs}, difference {diff}")
    return pair_counts


def get_het_pair_distribution(genotype_vector, masked_map, r_bins):
    """
    Get the number of site pairs per bin for an array of bins in r

    :param genotype_vector:
    :param masked_map:
    :param r_bins:
    :param max_i:
    :return:
    """
    n_bins = len(r_bins)
    n_het = genotype_vector.n_het
    pair_counts = np.zeros(n_bins, dtype=np.int64)
    d_bins = map_util.r_to_d(r_bins)  # convert bins in r to bins in d
    het_map = masked_map.values[genotype_vector.het_index]
    i = 0
    for i in np.arange(n_het):
        focus = het_map[i]
        focal_bins = d_bins + focus
        pos_bins = np.searchsorted(het_map, focal_bins)
        pos_bins[0, 0] = i + 1
        pair_counts += pos_bins[:, 1] - pos_bins[:, 0]
        #
        if i % 1e3 == 0:
            print(f"{i} bp scanned, {np.sum(pair_counts)} pairs binned")
    print(f"{i} bp scanned, {np.sum(pair_counts)} pairs binned")
    expected_n_pairs = int(n_het * (n_het - 1) / 2)
    n_pairs = int(np.sum(pair_counts))
    diff = int(n_pairs - expected_n_pairs)
    print(f"{n_pairs} recorded out of {expected_n_pairs}, difference {diff}")
    return pair_counts


def get_expected_pi_2(genotype_vector):
    """
    Compute pi_2 without bins

    :param genotype_vector:
    :param masked_map:
    :return:
    """
    n = genotype_vector.length
    n_hets = np.sum(genotype_vector.het_indicator)
    n_pairs = n * (n - 1) / 2
    n_het_pairs = n_hets * (n_hets - 1) / 2
    pi_2 = n_het_pairs / n_pairs
    return pi_2


# functions that take alt indicator vectors or other statistics as arguments

def compute_pi_x(alt_x, n_x):
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


def compute_pi_xy(alt_x, alt_y, n_x, n_y):
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

r_bins_1 = np.array([[0, 1e-8],
                     [1e-8, 2e-8], [2e-8, 5e-8], [5e-8, 1e-7],
                     [1e-7, 2e-7], [2e-7, 5e-7], [5e-7, 1e-6],
                     [1e-6, 2e-6], [2e-6, 5e-6], [5e-6, 1e-5],
                     [1e-5, 2e-5], [2e-5, 5e-5], [5e-5, 1e-4],
                     [1e-4, 2e-4], [2e-4, 5e-4], [5e-4, 1e-3],
                     [1e-3, 2e-3], [2e-3, 5e-3], [5e-3, 1e-2],
                     [1e-2, 2e-2], [2e-2, 5e-2], [5e-2, 1e-1],
                     [1e-1, 2e-1], [2e-1, 5e-1]], dtype=np.float32)

r_bins_2 = np.array([[0, 1e-7],
                     [1e-7, 2e-7], [2e-7, 5e-7], [5e-7, 1e-6],
                     [1e-6, 2e-6], [2e-6, 5e-6], [5e-6, 1e-5],
                     [1e-5, 2e-5], [2e-5, 5e-5], [5e-5, 1e-4],
                     [1e-4, 2e-4], [2e-4, 5e-4], [5e-4, 1e-3],
                     [1e-3, 2e-3], [2e-3, 5e-3], [5e-3, 1e-2],
                     [1e-2, 2e-2], [2e-2, 5e-2], [5e-2, 1e-1],
                     [1e-1, 2e-1], [2e-1, 5e-1]], dtype=np.float32)

r_bins_3 = np.array([[1e-6, 2e-6], [2e-6, 5e-6], [5e-6, 1e-5],
                     [1e-5, 2e-5], [2e-5, 5e-5], [5e-5, 1e-4],
                     [1e-4, 2e-4], [2e-4, 5e-4], [5e-4, 1e-3],
                     [1e-3, 2e-3], [2e-3, 5e-3], [5e-3, 1e-2]],
                    dtype=np.float32)

r_edges = 


_map = Map.load_txt(
    "c:/archaic/data/chromosomes/maps/chr22/genetic_map_GRCh37_chr22.txt")
_bed = Bed.load_bed(
    "c:/archaic/data/chromosomes/merged_masks/chr22/chr22_merge.bed")
maskedmap = MaskedMap.from_class(_map, _bed)
genotypevector = GenotypeVector.read_vcf(
    "c:/archaic/data/chromosomes/merged/chr22/complete_chr22.vcf.gz",
    "SS6004475")
