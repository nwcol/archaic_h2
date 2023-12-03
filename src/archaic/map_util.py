
# Utilities for representing and manipulating recombination maps

import numpy as np

import matplotlib

import matplotlib.pyplot as plt

from bed_util import Bed



class Map:
    """
    For loading .txt map files
    """

    def __init__(self, positions, rates, map_values):
        self.positions = positions
        self.rates = rates
        self.map_values = map_values

    @classmethod
    def load_txt(cls, path):
        """
        Assumes a file with a 1-line header, positions at column with index 1
        and map values in cM at column with index 3

        :param path:
        :return:
        """
        positions = []
        rates = []
        map_values = []
        with open(path, mode="r") as file:
            for i, line in enumerate(file):
                if i > 0:
                    fields = line.strip().split()
                    positions.append(fields[1])
                    rates.append(fields[2])
                    map_values.append(fields[3])
        positions = np.array(positions, dtype=np.int64)
        rates = np.array(rates, dtype=np.float64)
        map_values = np.array(map_values, dtype=np.float64)
        return cls(positions, rates, map_values)


class MaskedMap:
    """
    Class for combining a Map instance with a Bed instance, to obtain a vector
    of map coordinates for every position in the bed
    """

    def __init__(self, positions, map_values):
        self.positions = positions
        self.map_values = map_values

    @classmethod
    def from_class(cls, map, bed):
        bed_positions = bed.get_positions_1()
        approx_map_values = cls.approximate(bed_positions, map.positions,
                                            map.map_values, map.rates)
        return cls(bed_positions, approx_map_values)

    @classmethod
    def from_file(cls, map_path, bed_path):
        map = Map.load_txt(map_path)
        bed = Bed.load_bed(bed_path)
        return cls.from_class(map, bed)

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
        d = np.abs(self.map_values - self.map_values[i])
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


map = Map.load_txt(
    "c:/archaic/data/chromosomes/maps/chr22/genetic_map_GRCh37_chr22.txt")
bed = Bed.load_bed(
    "c:/archaic/data/chromosomes/merged_masks/chr22/chr22_merge.bed")
maskedmap = MaskedMap.from_class(map, bed)
