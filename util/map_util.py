
# Utilities for representing and manipulating recombination maps

import numpy as np
import matplotlib
import matplotlib.pyplot as plt


if __name__ == "__main__":
    plt.rcParams['figure.dpi'] = 100
    matplotlib.use('Qt5Agg')


class GeneticMap:

    def __init__(self, positions, values, chrom):
        self.positions = positions
        self.values = values
        self.chrom = chrom

    @classmethod
    def read_txt(cls, path):
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
