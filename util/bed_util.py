
# Bed class for loading and manipulating .bed files

import matplotlib.pyplot as plt
import matplotlib
import numpy as np
from util import vcf_util
from util import map_util


if __name__ == "__main__":
    plt.rcParams['figure.dpi'] = 100
    matplotlib.use('Qt5Agg')


class Bed:

    def __init__(self, regions, chrom):
        """
        Note that regions have 0-indexed starts and 1-indexed ends.

        Instances of this class are associated explicitly with single
        chromosomes by the chrom attribute
        """
        self.regions = regions
        self.chrom = chrom

    @classmethod
    def from_positions(cls, positions, chrom):
        """
        Initialize from a vector of 0-indexed positions.
        """
        high = np.max(positions)
        mask = np.zeros(high + 3, dtype=np.int32)
        mask[positions + 1] = 1
        dif = np.diff(mask)
        starts = np.where(dif == 1)[0]
        stops = np.where(dif == -1)[0]
        length = len(starts)
        regions = np.zeros((length, 2), dtype=np.int64)
        regions[:, 0] = starts
        regions[:, 1] = stops
        return cls(regions, chrom)

    @classmethod
    def read_vcf(cls, path):

        positions = vcf_util.read_positions(path) - 1
        chrom = str(vcf_util.read_chrom(path))
        return cls.from_positions(positions, chrom)

    @classmethod
    def read_bed(cls, path):

        starts = []
        stops = []
        with open(path, mode='r') as file:
            for i, line in enumerate(file):
                if "chromStart" not in line:
                    fields = line.split()
                    starts.append(int(fields[1]))
                    stops.append(int(fields[2]))
        chrom = fields[0]
        length = len(starts)
        regions = np.zeros((length, 2), dtype=np.int64)
        regions[:, 0] = starts
        regions[:, 1] = stops
        return cls(regions, chrom)

    @classmethod
    def read_map(cls, path):

        genetic_map = map_util.GeneticMap.read_txt(path)
        pos_0 = genetic_map.positions[0] - 1
        pos_1 = genetic_map.positions[-1]
        regions = np.array([[pos_0, pos_1]], dtype=np.int64)
        return cls(regions, genetic_map.chrom)

    def __getitem__(self, index):
        """
        Return a new instance holding the regions indexed by index
        """
        if isinstance(index, slice):
            index = range(*index.indices(self.n_regions))
        reg = self.regions[index]
        return Bed(reg, self.chrom)

    def trim(self, lower, upper):
        """
        Trim all regions below lower or above upper from the regions array
        """
        lower_index = np.searchsorted(self.regions[:, 0], lower)
        upper_index = np.searchsorted(self.regions[:, 1], upper)
        self.regions = self.regions[lower_index:upper_index]

    @property
    def n_regions(self):
        """
        Return the number of regions
        """
        return len(self.regions)

    @property
    def n_positions(self):
        """
        Return the total number of positions
        """
        return np.sum(self.lengths)

    @property
    def starts(self):
        """
        Return all start positions, 0-indexed
        """
        return self.regions[:, 0]

    @property
    def stops(self):
        """
        Return all stop positions, 1-indexed
        """
        return self.regions[:, 1]

    @property
    def lengths(self):
        """
        Return a vector of region lengths
        """
        return self.regions[:, 1] - self.regions[:, 0]

    @property
    def mean_length(self):
        """
        Return the mean region length
        """
        return np.mean(self.lengths)

    @property
    def median_length(self):
        """
        Return the median region length
        """
        return np.median(self.lengths)

    @property
    def max_length(self):
        """
        Return the maximum region length
        """
        return np.max(self.lengths)

    @property
    def min_pos(self):
        return self.regions[0, 0]

    @property
    def max_pos(self):
        """
        Return the maximum position covered, 1-indexed
        """
        return np.max(self.regions[:, 1])

    @property
    def min_length(self):
        """
        Return the size of the smallest region
        """
        return np.min(self.lengths)

    def exclude(self, limit):
        """
        Create a new instance excluding all regions with size smaller than
        limit
        """
        mask = np.nonzero(self.lengths >= limit)[0]
        regions = self.regions[mask]
        return Bed(regions, self.chrom)

    def plot_coverage(self, min_length, max_pos=None):
        """
        Plot the coverage of tracts above a minimum size on the chromosome
        """
        fig = plt.figure(figsize=(12, 3))
        sub = fig.add_subplot(111)
        lengths = self.lengths
        large = np.where(lengths > min_length)[0]
        if not max_pos:
            max_pos = self.max_pos
        sub.barh(y=0, width=max_pos, color="white", edgecolor="black")
        sub.barh(y=0, width=lengths[large], left=self.starts[large],
                 color="blue")
        sub.set_ylim(-1, 1)
        sub.set_xlim(-1e6, max_pos + 1e6)
        sub.set_yticks(ticks=[0], labels=["chr22"])
        sub.set_xlabel("position")
        fig.show()

    def plot_distribution(self, bins=50, y_max=None):
        """
        Plot the distribution of region lengths
        """
        fig = plt.figure(figsize=(6, 6))
        sub = fig.add_subplot(111)
        sub.hist(self.lengths, bins=bins, color="blue")
        sub.set_xlim(0, )
        if y_max:
            sub.set_ylim(0, y_max)
        sub.set_ylabel("n regions")
        sub.set_xlabel("region size, bp")
        fig.show()

    def get_0_idx_positions(self):
        """
        Return a vector of 0-indexed positions for all regions
        """
        mask = self.get_position_mask()
        positions = np.nonzero(mask == 1)[0]
        return positions

    def get_1_idx_positions(self):
        """
        Return a vector of .vcf-style, 1-indexed positions for all regions
        """
        mask = self.get_position_mask()
        positions = np.nonzero(mask == 1)[0]
        positions += 1
        return positions

    def get_position_mask(self, max_pos=None):
        """
        Return a vector holding 1s where positions exist in the Bed instance.
        1-indexed
        """
        if not max_pos:
            max_pos = self.max_pos
        mask = np.zeros(max_pos, dtype=np.uint8)
        for start, stop in self.regions:
            mask[start:stop] = 1
        return mask

    def write_bed(self, path):
        """
        Save the regions array as a .bed file.
        """
        with open(path, 'w') as file:
            for i in np.arange(self.n_regions):
                file.write(
                    "\t".join(
                        [
                            self.chrom,
                            str(self.regions[i, 0]),
                            str(self.regions[i, 1])
                        ]
                    ) + "\n"
                )
        print(f".bed file for chr {self.chrom} with {self.n_positions} "
              f"positions written at {path}")


def intersect_beds(*beds):

    n_beds = len(beds)
    chrom = beds[0].chrom
    maximum = max([bed.max_pos for bed in beds])
    masks = [bed.get_position_mask(max_pos=maximum) for bed in beds]
    tally = np.sum(masks, axis=0)
    overlaps = np.where(tally == n_beds)[0]
    intersect = Bed.from_positions(overlaps, chrom)
    return intersect
