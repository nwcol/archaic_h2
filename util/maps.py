
"""
A class for representing and manipulating recombination maps
"""

import numpy as np
import matplotlib
import matplotlib.pyplot as plt


if __name__ == "__main__":
    plt.rcParams['figure.dpi'] = 100
    matplotlib.use('Qt5Agg')


data_path = "/home/nick/Projects/archaic/data"


class GeneticMap:

    def __init__(self, positions, map_vals):
        """
        Positions are assumed to be 1-indexed.

        :param positions:
        :param map_vals:
        :param chrom:
        """
        self.positions = positions
        self.map_vals = map_vals
        self.n_points = len(map_vals)

    @classmethod
    def read_txt(cls, file_name, pos_col=None, map_col=None):
        """
        Read a map from a .txt file. Designed to read the hapmap format;

        Chromosome\tPosition(bp)\t(Rate(cM/Mb)\tMap(cM)
        """
        pos_idx = None
        map_idx = None
        positions = []
        map_vals = []
        with open(file_name, mode="r") as file:
            lines = iter(file)
            header = next(lines)
            header_fields = header.strip("\n").split()
            header_fields = [x.strip('"') for x in header_fields]
            if pos_col:
                if pos_col in header_fields:
                    pos_idx = header_fields.index(pos_col)
                else:
                    raise ValueError(f"pos_col {pos_col} isn't in header!")
            else:
                for x in ["Position(bp)", "Physical_Pos"]:
                    if x in header_fields:
                        pos_idx = header_fields.index(x)
                        break
                if pos_idx is None:
                    raise ValueError("no position column found!")
            if map_col:
                if map_col in header_fields:
                    map_idx = header_fields.index(map_col)
                else:
                    raise ValueError(f"map_col {map_col} isn't in header!")
            else:
                for x in ["Map(cM)"]:
                    if x in header_fields:
                        map_idx = header_fields.index(x)
                        break
                if pos_idx is None:
                    raise ValueError("no map column found!")
            for line in lines:
                fields = line.strip("\n").split()
                positions.append(fields[pos_idx])
                map_vals.append(fields[map_idx])
        positions = np.array(positions, dtype=np.int64)
        map_vals = np.array(map_vals, dtype=np.float64)
        return cls(positions, map_vals)

    @property
    def first_position(self):
        """
        Return the first position covered by the map
        """
        return self.positions[0]

    @property
    def last_position(self):
        """
        Return the last position covered by the map
        """
        return self.positions[-1]

    def approximate_map_rates(self):
        """
        Return approximate map rates in units of cM/bp.
        """
        approx_rates = np.diff(self.map_vals) / np.diff(self.positions)
        approx_rates = np.append(approx_rates, 0)
        return approx_rates

    def approximate_map_value(self, position):
        return self.approximate_map_values(np.array([position]))[0]

    def approximate_map_values(self, positions):
        """
        Approximate the map values of a vector of positions between
        self.first_position and self.last_position

        :param positions:
        :return:
        """
        floor_idx = np.searchsorted(self.positions, positions, "right") - 1
        floor_idx[floor_idx < 0] = 0
        floor = self.map_vals[floor_idx]
        floor_positions = self.positions[floor_idx]
        bp_to_floor = positions - floor_positions
        approx_rates = self.approximate_map_rates()
        position_rates = approx_rates[floor_idx]
        approx_map = np.zeros(len(positions), dtype=np.float64)
        approx_map[:] = floor + bp_to_floor * position_rates
        return approx_map

    def find_interval_end(self, focal_pos, r):
        # 1 indexed positions I think. returns the first position ABOVE r
        # (noninclusive)
        focal_map_val = self.approximate_map_value(focal_pos)
        approx_val = focal_map_val + r_to_d(r)
        idx = np.searchsorted(self.map_vals, approx_val)
        if idx == len(self.positions):
            idx -= 1
        positions = np.arange(self.positions[idx - 1], self.positions[idx])
        map_vals = self.approximate_map_values(positions)
        map_distances = np.abs(map_vals - focal_map_val)
        r_distances = d_to_r(map_distances)
        out_idx = np.searchsorted(r_distances, r)
        if out_idx < len(positions):
            position = positions[out_idx]
        else:
            position = positions[-1]
        return position

    def plot_rates(self, log_scale=False):

        fig = plt.figure(1)
        ax = fig.add_subplot(111)
        scaled_positions = self.positions / 1e6
        ax.plot(scaled_positions, self.map_rates, color="black")
        if log_scale:
            ax.set_yscale("log")
        ax.set_xlim(self.first_position / 1e6, self.last_position / 1e6)
        ax.set_xlabel("position, Mb")
        ax.set_ylabel("cM/Mb")
        fig.tight_layout()
        fig.show()

    def plot_map(self):

        fig = plt.figure(1)
        ax = fig.add_subplot(111)
        scaled_positions = self.positions
        ax.plot(scaled_positions, self.map_vals, color="black")
        ax.set_xlim(self.first_position, self.last_position)
        ax.set_xlabel("position")
        ax.set_ylabel("cM")
        ax.grid(alpha=0.2)
        ax.set_ylim(0, )
        fig.tight_layout()
        fig.show()

    def plot_r_map(self, focal_pos):

        fig = plt.figure(1)
        ax = fig.add_subplot(111)
        focal_map_val = self.approximate_map_values(np.array([focal_pos]))[0]
        map_distance = np.abs(self.map_vals - focal_map_val)
        r_distance = d_to_r(map_distance)
        ax.plot(self.positions, r_distance, color="black")
        ax.set_xlim(self.first_position, self.last_position)
        ax.set_xlabel("position")
        ax.set_ylabel("r")
        ax.grid(alpha=0.2)
        ax.set_ylim(0, )
        fig.tight_layout()
        fig.show()

    def compare(self):

        edges = np.linspace(0, 20_000, 21, dtype=np.int64)
        out = np.zeros((self.n_points, len(edges)))
        for i in range(self.n_points):
            pos = self.positions[i]
            for k in range(len(edges) - 1):
                idx = np.where(
                    (self.positions > pos + edges[k])
                    & (self.positions <= pos + edges[k+1])
                    )
                out[i, k] = np.mean(self.map_vals[idx])
        out = d_to_r(out)
        return out


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
    :return:
    """
    d_distance = -50 * np.log(1 - 2 * r_distance)
    return d_distance