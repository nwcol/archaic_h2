import matplotlib.pyplot as plt

import matplotlib

import numpy as np

import time

import os

import sys


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