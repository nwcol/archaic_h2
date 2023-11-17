
# Bed class for loading and manipulating .bed files

import gzip

import matplotlib.pyplot as plt

import matplotlib

import numpy as np

import vcf_util


if __name__ == "__main__":
    plt.rcParams['figure.dpi'] = 100
    matplotlib.use('Qt5Agg')


class Bed:

    def __init__(self, regions, chrom):
        """
        Note that regions have 0-indexed starts and 1-indexed ends.

        Instances of this class are associated explicitly with single
        chromosomes by the chrom attribute

        :param regions: 2d array of region starts/stops. shape (l, 2)
        :type regions: np.ndarray
        :param chrom: names the associated chromosome
        :type chrom: str
        """
        self.regions = regions
        self.chrom = chrom

    @classmethod
    def new(cls, file_name):
        """
        Instantiate from a .vcf.gz file by recording all continuous regions

        :param file_name: .vcf.gz file name
        :type file_name: str
        :return: class instance
        """
        starts = []
        stops = []
        pos0 = -1
        starts.append(-1)
        with gzip.open(file_name, mode="r") as file:
            for i, line in enumerate(file):
                if b'#' not in line:
                    pos = vcf_util.parse_position(line)
                    if pos - pos0 > 1:
                        starts.append(pos)
                        stops.append(pos0)
                    pos0 = pos
            stops.append(pos)
        length = len(starts) - 1
        regions = np.zeros((length, 2), dtype=np.int64)
        regions[:, 0] = starts[1:]
        regions[:, 0] -= 1  # set starts to 0 index
        regions[:, 1] = stops[1:]
        chrom = vcf_util.parse_chrom(file_name)
        return cls(regions, chrom)

    @classmethod
    def load(cls, file_name):
        """
        Instantiate from an existing .bed file

        :param file_name:
        :return:
        """
        starts = []
        stops = []
        with open(file_name, mode='r') as file:
            for i, line in enumerate(file):
                if i > 1:
                    chrom, start, stop = line.split()
                    starts.append(int(start))
                    stops.append(int(stop))
        length = len(starts)
        regions = np.zeros((length, 2), dtype=np.int64)
        regions[:, 0] = starts
        regions[:, 1] = stops
        self = cls(regions, chrom)

    def __len__(self):
        """
        Return the number of regions recorded in self.regions, which is the
        same as the length of that array.

        :return:
        """
        return len(self.regions)

    def __getitem__(self, index):
        """
        Return a new instance holding the regions indexed by index

        :param index:
        :return:
        """
        if isinstance(index, slice):
            index = range(*index.indices(len(self)))
        reg = self.regions[index]
        return Bed(reg, self.chrom)

    @property
    def starts(self):
        """
        Return all start positions

        :return:
        """
        return self.regions[:, 0]

    @property
    def stops(self):
        """
        Return all stop positions

        :return:
        """
        return self.regions[:, 1]

    @property
    def length(self):
        """
        Return the length of the self.regions array

        :return:
        """
        return len(self.regions)

    @property
    def lengths(self):
        """
        Return a vector of region lengths

        :return:
        """
        return self.regions[:, 1] - self.regions[:, 0]

    @property
    def mean_length(self):
        """
        Return the mean region length

        :return:
        """
        return np.mean(self.lengths)

    @property
    def median_length(self):
        """
        Return the median region length

        :return:
        """
        return np.median(self.lengths)

    @property
    def sum(self):
        """
        Return the sum of all region lengths

        :return:
        """
        return np.sum(self.lengths)

    @property
    def max_length(self):
        """
        Return the maximum region length

        :return:
        """
        return np.max(self.lengths)

    @property
    def max_pos(self):
        """
        Return the maximum position covered.

        :return:
        """
        return np.max(self.regions[:, 1]) - 1

    @property
    def min_length(self):
        return np.min(self.lengths)

    def coverage(self, end=None):
        if not end:
            end = self.max_pos
        return self.sum / end

    def exclude(self, limit):
        """
        Create a new instance excluding all regions smaller than 'limit'

        :param limit:
        :return:
        """
        mask = np.nonzero(self.lengths > limit)[0]
        regions = self.regions[mask]
        return Bed(regions, self.chrom)

    def plot_coverage(self, min_length, chrom_length=None):
        """
        Plot the coverage of tracts above a minimum size on the chromosome

        :param min_length:
        :return:
        """
        fig = plt.figure(figsize=(12, 3))
        sub = fig.add_subplot(111)
        lengths = self.lengths
        large = np.where(lengths > min_length)[0]
        if not chrom_length:
            chrom_length = self.max_pos
        sub.barh(y=0, width=chrom_length, color="white", edgecolor="black")
        sub.barh(y=0, width=lengths[large], left=self.starts[large], color="blue")
        sub.set_ylim(-1, 1)
        sub.set_xlim(-1e6, chrom_length + 1e6)
        sub.set_yticks(ticks=[0], labels=["chr22"])
        sub.set_xlabel("position")
        sub.set_title(f"approx coverage {np.round(self.coverage(), 3)}")
        fig.show()

    def plot_distribution(self, bins=50, ymax=None):
        """
        Plot the distribution of region lengths

        :param bins:
        :return:
        """
        fig = plt.figure(figsize=(6, 6))
        sub = fig.add_subplot(111)
        sub.hist(self.lengths, bins=bins, color="blue")
        sub.set_xlim(0, )
        if ymax:
            sub.set_ylim(0, ymax)
        sub.set_ylabel("n regions")
        sub.set_xlabel("region size, bp")
        fig.show()

    def write_bed(self, file_name):
        """
        Save the regions array as a .bed file

        :param file_name:
        :type file_name: str
        :return:
        """
        with open(file_name, 'w') as file:
            for i in np.arange(self.length):
                file.write("\t".join([self.chrom,
                                      str(self.regions[i, 0]),
                                      str(self.regions[i, 1])])
                           + "\n")
        return 0
