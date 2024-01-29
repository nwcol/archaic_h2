
# Utilities for representing and manipulating recombination maps

import numpy as np

import matplotlib

import matplotlib.pyplot as plt

import sys

from util import bed_util


if __name__ == "__main__":
    plt.rcParams['figure.dpi'] = 100
    matplotlib.use('Qt5Agg')


class GeneticMap:

    def __init__(self, positions, values, chrom):
        self.positions = positions
        self.values = values
        self.chrom = chrom

    @classmethod
    def load_txt(cls, path):
        """
        Assumes a file with a 1-line header, positions at column with index 1
        and map values in cM at column with index 3

        :param path:
        :return:
        """
        positions = []
        values = []
        with open(path, mode="r") as file:
            for i, line in enumerate(file):
                if i > 0:
                    fields = line.strip().split()
                    positions.append(fields[1])
                    values.append(fields[3])
        chrom = fields[0].strip("chr")
        positions = np.array(positions, dtype=np.int64)
        values = np.array(values, dtype=np.float64)
        return cls(positions, values, chrom)

    @property
    def first_pos(self):
        return self.positions[0]

    @property
    def last_pos(self):
        return self.positions[-1]

    def approximate_map_values(self, positions):
        index = np.searchsorted(self.positions, positions, "right") - 1
        floor = self.values[index]
        floor_positions = self.positions[index]
        bp_to_floor = positions - floor_positions
        approx_rates = np.diff(self.values) / np.diff(self.positions)
        approx_rates = np.append(approx_rates, 0)  # in case last site exists
        pos_approx_rates = approx_rates[index]
        approx_map = np.zeros(len(positions), dtype=np.float64)
        approx_map[:] = floor + bp_to_floor * pos_approx_rates
        return approx_map


class MaskedMap:
    """
    Class for combining a Map instance with a Bed instance, to obtain a vector
    of map coordinates for every position in the bed
    """

    def __init__(self, positions, values):
        self.positions = positions
        self.values = values
        self.length = len(self.positions)

    @classmethod
    def from_class(cls, map, bed):
        positions = bed.get_positions_1()
        approx_map_values = cls.approximate_map(bed, map)
        return cls(positions, approx_map_values)

    @classmethod
    def from_file(cls, map_path, bed_path):
        map = Map.load_txt(map_path)
        bed = Bed.load_bed(bed_path)
        return cls.from_class(map, bed)

    @staticmethod
    def approximate_map(bed, map):
        """
        Approximate the map values for a vector of vcf positions using the
        map values and rates from a recombination map

        assume map file is 1 indexed

        :param bed_positions:
        :param map_positions:
        :param map_values:
        :param rates:
        :return:
        """

        n_positions = bed.n_positions
        positions = bed.get_positions_1()
        index = np.searchsorted(map.positions, positions, "right") - 1
        floor = map.values[index]
        floor_positions = map.positions[index]
        bp_to_floor = positions - floor_positions
        #Mb_to_floor = bp_to_floor * 1e-6
        approx_rates = np.diff(map.values) / np.diff(map.positions)# * 1e6
        pos_approx_rates = approx_rates[index]
        approx_map = np.zeros(n_positions, dtype=np.float64)
        approx_map[:] = floor + bp_to_floor * pos_approx_rates
        return approx_map

    @staticmethod
    def approximate(bed_positions, map_positions, map_values, rates):
        """
        Approximate the map values for a vector of vcf positions using the
        map values and rates from a recombination map

        :param bed_positions:
        :param map_positions:
        :param map_values:
        :param rates:
        :return:
        """
        index_in_map = np.searchsorted(map_positions, bed_positions) - 1
        floor = map_values[index_in_map]
        floor_positions = map_positions[index_in_map]
        bp_to_floor = bed_positions - floor_positions
        Mb_to_floor = bp_to_floor * 1e-6
        pos_rates = rates[index_in_map]
        approx_map_values = floor + Mb_to_floor * pos_rates
        return approx_map_values

    def compute_r(self, i):
        """
        Compute r between a position and all other sites. The position must
        be included in the positions vector. Uses Haldanes map function

        :param pos:
        :return:
        """
        d = np.abs(self.values - self.values[i])
        r = 0.5 * (1 - np.exp(-0.02 * d))
        return r
    
    def sites_in_r(self, i, r_0, r_1):
        """
        Return the indices of sites within range r greater than r_0 and less
        than r_1 of position indexed by i
        
        :param i: position index
        :param r_0: lower r bound
        :param r_1: upper r bound
        :return: 
        """
        r_about_i = self.compute_r(i)
        sites = np.nonzero((r_about_i > r_0) & (r_about_i <= r_1))[0]
        return sites

    def approximate_r_bins(self, i, d_bins):
        """
        Only searches UPWARDS
        bounds are [, ), so subtracting column 0 from 1 gives length

        :param r_bounds:
        :return:
        """
        start = self.values[i]
        bins = d_bins + start
        pos_bins = np.searchsorted(self.values, bins)
        pos_bins[0, 0] = i + 1
        return pos_bins


def count_pairings(maskedmap, d_bins):
    sums = np.zeros(len(d_bins), dtype=np.int64)
    n = maskedmap.length
    for i in np.arange(n):
        counts = maskedmap.approximate_r_bins(i, d_bins)
        sums += counts[:, 1] - counts[:, 0]
    return sums


def d_to_r(d_distance):
    """
    Use Haldane's map function to convert map units cM to recombination
    frequency r

    Examples:
    d_to_r(200) = 0.4908421805556329
    d_to_r(30) = 0.22559418195298675
    d_to_r(1) = 0.009900663346622374
    d_to_r(0.5) = 0.004975083125415947

    :param d_distance: distance given in map units
    :type d_distance: float or array of floats
    :return:
    """
    r_distance = 0.5 * (1 - np.exp(-0.02 * d_distance))
    return r_distance


def r_to_d(r_distance):
    """
    Use Haldane's map function to convert recombination frequency r to map
    units cM

    Examples:
    r_to_d(0.49) = 195.60115027140725
    r_to_d(0.01) = 1.0101353658759733 about one Mb
    r_to_d(0.0001) = 0.010001000133352235 about one kb
    r_to_d(1e-8) = 1.0000000094736444e-06 about one bp

    :param r_distance: recombination frequency r
    :type r_distance: float or array of floats
    :return:
    """
    d_distance = -50 * np.log(1 - 2 * r_distance)
    return d_distance
