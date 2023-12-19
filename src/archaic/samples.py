import numpy as np

import vcf_util


"""
general to do

-renaming samples: how to handle
-making a nice file mapping sample labels to each otehr
-set this class up; combine multiple samples, approximated map, etc
    and bed stuff?
-being nice to .vfs: keeping the headers intact etc

"""


class GeneticMap:

    def __init__(self, positions, values):
        self.positions = positions
        self.values = values

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
        positions = np.array(positions, dtype=np.int64)
        values = np.array(values, dtype=np.float64)
        return cls(positions, values)

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
        pos_approx_rates = approx_rates[index]
        approx_map = np.zeros(len(positions), dtype=np.float64)
        approx_map[:] = floor + bp_to_floor * pos_approx_rates
        return approx_map


class Samples:

    def __init__(self, positions, samples):
        """

        :param positions:
        :param samples: dict {sample_id: genotypes...}
        """
        self.positions = positions
        self.n_positions = len(positions)
        self.samples = samples
        self.sample_ids = samples.keys()
        self.n_samples = len(samples)
        self.map_values = None

    @classmethod
    def load_vcf_slow(cls, path):
        """
        loop over each sample individually

        :param path:
        :return:
        """
        n_positions = vcf_util.count_positions(path)
        positions = vcf_util.read_positions(path, n_positions)
        labels = vcf_util.read_sample_ids(path)
        samples = {label.decode(): None for label in labels}
        for label in samples:
            samples[label] = vcf_util.read_sample(path, label, n_positions)
        return cls(positions, samples)

    @classmethod
    def load_vcf(cls, path):
        samples, positions = vcf_util.read_samples(path)
        return cls(positions, samples)

    @classmethod
    def load___(cls):
        pass

    def add_sample(self, label, positions, genotypes):
        if np.sum(positions == positions) != self.n_positions:
            raise ValueError("")
        self.n_samples += 1
        self.samples[label] = genotypes

    def get_map(self, genetic_map):
        self.map_values = genetic_map.approximate_map_values(self.positions)









