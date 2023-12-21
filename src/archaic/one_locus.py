import matplotlib.pyplot as plt

import matplotlib

import numpy as np

import time

import os

import sys

import vcf_samples


if __name__ == "__main__":
    plt.rcParams['figure.dpi'] = 100
    matplotlib.use('Qt5Agg')


def compute_pi(samples, *sample_ids):
    """
    Compute diversity for one sample in a Samples index

    :param samples:
    :param sample_ids: sample ids to compute pi for
    :return:
    """
    n = len(sample_ids) * 2
    alts = [samples.alts(sample_id) for sample_id in sample_ids]
    alt_sum = np.sum(alts, axis=0)
    ref_sum = n - alt_sum
    tot = alt_sum * ref_sum
    pi = 2 / (samples.n_positions * n * (n - 1)) * np.sum(tot)
    return pi


def compute_h(samples, sample_id):
    """
    Compute heterozygosity for one sample in a Samples index

    :param samples:
    :param sample_id:
    :return:
    """
    h = samples.n_hets(sample_id) / samples.n_positions
    return h


def compute_pi_xy(samples, id_x, id_y):
    """
    Compute divergence between two samples

    :param samples:
    :param id_x:
    :param id_y:
    :return:
    """
    n = 2
    alt_x = samples.alts(id_x)
    alt_y = samples.alts(id_y)
    ref_x = n - alt_x
    ref_y = n - alt_y
    tot = (ref_x * alt_y) + (alt_x * ref_y)
    pi_xy = 1 / (samples.n_positions * n * n) * np.sum(tot)
    return pi_xy


def compute_all_pi(samples):
    """
    Compute diversity for each sample in a Samples instance

    :param samples:
    :return:
    """
    pi_dict = {sample_id: None for sample_id in samples.sample_ids}
    for sample_id in pi_dict:
        pi_dict[sample_id] = compute_pi(samples, sample_id)
    return pi_dict


def compute_all_pi_xy(samples):
    """
    Compute divergence between all pairs of samples

    :param samples:
    :return:
    """
    n_samples = samples.n_samples
    sample_ids = samples.sample_ids
    pi_xy_matrix = np.zeros((n_samples, n_samples), dtype=np.float64)
    for i in np.arange(n_samples):
        for j in np.arange(i + 1, n_samples):
            x = sample_ids[i]
            y = sample_ids[j]
            pi_xy_matrix[i, j] = compute_pi_xy(samples, x, y)
    return pi_xy_matrix, sample_ids































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
    def read_vcf(cls, vcf_path, sample, bed=None):
        if not bed:
            bed = Bed.from_vcf(vcf_path)
        positions = bed.get_positions_1()
        n_positions = bed.n_positions
        genotypes = vcf_util.read_sample(vcf_path, sample, n_positions)
        return cls(positions, genotypes)


    @classmethod
    def read_abbrev_vcf(cls, vcf_path, bed_path, sample):
        """
        Read a simulated .vcf which records only alternate alleles into the
        positions defined in a .bed file

        :param vcf_path:
        :param bed_path:
        :param sample:
        :return:
        """
        bed = Bed.load_bed(bed_path)
        positions = bed.get_positions_1()
        n_positions = bed.n_positions
        genotypes = np.zeros(n_positions, dtype=np.uint8)
        abbrev_genotypes = vcf_util.read_sample(vcf_path, sample)
        abbrev_positions = vcf_util.read_positions(vcf_path)
        index = np.searchsorted(positions, abbrev_positions)
        genotypes[index] = abbrev_genotypes
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


# functions that take alt indicator vectors or other statistics as arguments

def compute_pi_xxx(alt_x, n_x):
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


def compute_pi_xyxx(alt_x, alt_y, n_x, n_y):
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


samples = vcf_samples.Samples.dir("c:/archaic/data/chromosomes/merged/chr22/")