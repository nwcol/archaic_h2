
# Functions for computing one-locus genetic statistics

import matplotlib.pyplot as plt
import matplotlib
import numpy as np
from util import sample_sets
from util import vcf_util


if __name__ == "__main__":
    plt.rcParams['figure.dpi'] = 100
    matplotlib.use('Qt5Agg')


def compute_pi(sample_set, *sample_ids):
    """
    Compute nucleotide diversity
    """
    if len(sample_ids) == 0:
        sample_ids = sample_set.sample_ids
    L = sample_set.n_positions
    pi_dict = {sample_id: 0 for sample_id in sample_ids}
    for sample_id in pi_dict:
        pi_dict[sample_id] = sample_set.n_het(sample_id) / L
    return pi_dict


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
