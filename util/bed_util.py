
"""
A class for loading and manipulating .bed files
"""

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib
import numpy as np
from util import vcf_util
from util import map_util


data_path = "/home/nick/Projects/archaic/data"


if __name__ == "__main__":
    plt.rcParams['figure.dpi'] = 100
    matplotlib.use('Qt5Agg')


class Bed:

    columns = [
        "chrom",
        "chromStart",
        "chromEnd",
        "name",
        "score",
        "strand",
        "thickStart",
        "thickEnd",
        "itemRgb",
        "blockCount",
        "blockSizes",
        "blockStarts"
    ]

    def __init__(self, regions, chroms, names=None, scores=None):

        self.regions = regions
        self.chroms = chroms
        self.unique_chroms = list(set(chroms))
        #
        if len(self.unique_chroms) == 1:
            self.chrom = self.unique_chroms[0]
        else:
            self.chrom = None
        #
        self.names = names or None
        self.scores = scores or None

    """
    Instantiation from existing structures
    """

    @classmethod
    def from_positions_1(cls, positions_1, chroms):
        """
        Initialize from a vector of 1-indexed positions.
        """
        rightmost_position = np.max(positions_1)
        # length is incremented by 2 so that the 0th and -1st elements are 0
        indicator = np.zeros(rightmost_position + 2, dtype=np.int8)
        indicator[positions_1] = 1
        jumps = np.diff(indicator)
        starts = np.where(jumps == 1)[0]
        stops = np.where(jumps == -1)[0]
        length = len(starts)
        regions = np.zeros((length, 2), dtype=np.int64)
        regions[:, 0] = starts
        regions[:, 1] = stops
        if len(chroms) != length:
            chroms = np.full(length, chroms[0])
        return cls(regions, chroms)

    @classmethod
    def from_positions_0(cls, positions_0, chroms):
        """
        Initialize from a vector of 0-indexed positions
        """
        positions_1 = positions_0 + 1
        return cls.from_positions_1(positions_1, chroms)

    @classmethod
    def from_boolean_mask_0(cls, mask, chroms):
        """
        Initialize from a 0-indexed boolean mask
        """
        positions_0 = np.nonzero(mask)[0]
        positions_1 = positions_0 + 1
        return cls.from_positions_1(positions_1, chroms)

    """
    Loading from file
    """

    @classmethod
    def read_vcf(cls, file_name):
        """
        Create a mask from the positions in a .vcf.gz file
        """
        positions_1 = vcf_util.read_positions(file_name)
        chroms = vcf_util.read_chrom_col(file_name)
        return cls.from_positions_1(positions_1, chroms)

    @classmethod
    def read_bed(cls, file_name):
        """
        Read a .bed file
        """
        with open(file_name, mode='r') as file:
            for line in file:
                line_0 = line.strip('\t').split('\t')
                break
        start, end = None, None
        if line_0[1].isnumeric():
            n_cols = len(line_0)
            col_names = cls.columns[:n_cols]
            has_header = False
            start = "chromStart"
            end = "chromEnd"
        else:
            col_names = line_0
            has_header = True
            for name in col_names:
                if "start" in name or "Start" in name:
                    if not start:
                        start = name
                elif "end" in name or "End" in name:
                    if not end:
                        end = name
        fields = {name: [] for name in col_names}
        #
        with open(file_name, mode='r') as file:
            for i, line in enumerate(file):
                if i >= has_header:
                    line_fields = line.rstrip("\n").split("\t")
                    for j, field in enumerate(line_fields):
                        if field.isnumeric():
                            field = int(field)
                        elif "chr" in field:
                            field = int(field.lstrip("chrom"))
                        fields[col_names[j]].append(field)
        #
        chroms = np.array(fields[col_names[0]])
        regions = np.array([fields[start], fields[end]], dtype=np.int64).T
        return cls(regions, chroms)

    @classmethod
    def read_map(cls, file_name):
        """
        Get a Bed instance with a single region representing the coverage of
        a genetic map
        """
        genetic_map = map_util.GeneticMap.read_txt(file_name)
        pos = genetic_map.positions
        start_1 = pos[0]
        stop_1 = pos[-1]
        regions = np.array([[start_1 - 1, stop_1]], dtype=np.int64)
        return cls(regions, [genetic_map.chrom])

    @classmethod
    def read_chr(cls, chrom):
        """
        Just an abbreviated way to load a single .bed
        """
        path = f"{data_path}/masks/chr{chrom}.bed"
        return cls.read_bed(path)

    """
    Subset new masks from existing mask
    """

    def flank(self, distance):
        """
        Return a new bed with added flanking coverage
        """
        regions = self.regions
        indicator = np.zeros(regions[-1, 1] + distance, dtype=np.int64)
        for reg in regions:
            lower = max(reg[0] - distance, 0)
            upper = reg[1] + distance
            indicator[lower:upper] += 1
        mask = indicator > 0
        new_bed = Bed.from_boolean_mask_0(mask, self.chroms)
        return new_bed

    def subset_chrom(self, chrom):
        """
        Get a new instance containing only regions on a given chromosome
        """
        mask = self.get_chrom_mask(chrom)
        regions = self.regions[mask]
        chroms = self.chroms[mask]
        return Bed(regions, chroms)

    def exclude(self, limit):
        """
        Create a new instance excluding all regions with size smaller than
        'limit'
        """
        mask = np.nonzero(self.lengths >= limit)[0]
        regions = self.regions[mask]
        chroms = self.chroms[mask]
        return Bed(regions, chroms)

    def parse(self):
        """
        merge overlapping regions
        """
        regions = self.regions
        indicator = np.zeros(self.last_position, dtype=np.int64)
        for start, stop in regions:
            indicator[start:stop] = 1
        mask = indicator > 0
        chroms = [self.chrom]
        return Bed.from_boolean_mask_0(mask, chroms)

    """
    Mask vectors
    """

    def get_chrom_mask(self, chrom):
        """
        Return a boolean mask for a specific chromosome
        """
        mask = self.chroms == chrom
        return mask

    def get_sized_indicator(self, length=None):
        """
        Return a vector holding 1s where positions exist in the Bed instance.
        0-indexed
        """
        if not length:
            length = self.last_position
        indicator = np.zeros(length, dtype=np.int8)
        for start, stop in self.regions:
            indicator[start:stop] = 1
        return indicator

    def get_sized_boolean_mask(self, length=None):
        """
        Get a 0-indexed boolean mask of given length
        """
        mask = self.get_sized_indicator(length=length) == 1
        return mask

    """
    Properties
    """

    @property
    def n_positions(self):
        return np.sum(self.lengths)

    @property
    def lengths(self):
        return self.regions[:, 1] - self.regions[:, 0]

    @property
    def n_regions(self):
        return len(self.regions)

    @property
    def first_position(self):
        return self.regions.min()

    @property
    def last_position(self):
        return self.regions.max()

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
    def positions_0(self):
        """
        Return a vector of 0-indexed positions for all regions

        This is a property rather than an attribute so that it doesn't take up
        memory
        """
        positions = np.nonzero(self.indicator)[0]
        return positions

    @property
    def positions_1(self):
        """
        Return a vector of .vcf-style, 1-indexed positions for all regions
        """
        positions = np.nonzero(self.indicator)[0]
        positions += 1
        return positions

    @property
    def indicator(self):
        """
        Return a vector of length self.last_position holding 1 wherever
        a position is covered. 0-indexed
        """
        indicator = np.zeros(self.last_position, dtype=np.int8)
        for start, stop in self.regions:
            indicator[start:stop] = 1
        return indicator

    @property
    def boolean_mask(self):
        """
        Return a 0-indexed boolean mask of covered positions
        """
        mask = self.indicator == 1
        return mask

    @property
    def flat_regions(self):
        """
        Return a vector of ravelled region start/stop positions
        """
        return np.ravel(self.regions)

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
    def min_length(self):
        """
        Return the size of the smallest region
        """
        return np.min(self.lengths)

    @property
    def max_length(self):
        """
        Return the maximum region length
        """
        return int(np.max(self.lengths))

    @property
    def score_density(self):

        densities = np.zeros(self.last_position, dtype=np.int32)
        for i, [start, stop] in enumerate(self.regions):
            densities[start:stop] = self.scores[i]
        return densities

    """
    Windows
    """

    def subset_regions(self, window):
        """
        Return a subset of the regions array within a window. If a region
        crosses the upper

        :param window:
        """
        start, stop = window
        low = np.searchsorted(self.stops, start)
        high = np.searchsorted(self.starts, stop)
        n_regions = len(self.regions[low:high])
        window_regions = np.zeros((n_regions, 2), dtype=np.int64)
        window_regions[:, :] = self.regions[low:high]
        window_regions[window_regions < start] = start
        window_regions[window_regions >= stop] = stop - 1
        return window_regions

    def count_positions(self, window):

        return np.diff(np.searchsorted(self.positions_1, window))

    def get_window_position_count(self, window):

        count = np.searchsorted(self.positions_1, window[1])\
              - np.searchsorted(self.positions_1, window[0])
        return int(count)

    def get_vec_window_position_count(self, windows):

        counts = np.searchsorted(self.positions_1, windows[:, 1])\
               - np.searchsorted(self.positions_1, windows[:, 0])
        return counts

    def get_window_coverage(self, window):

        count = self.get_window_position_count(window)
        span = window[1] - window[0]
        return count / span

    def get_window_coverages(self, window_arr):
        # vectorized
        counts = np.searchsorted(self.positions_1, window_arr[:, 1])\
               - np.searchsorted(self.positions_1, window_arr[:, 0])
        spans = window_arr[:, 1] - window_arr[:, 0]
        return counts / spans

    """
    Plots
    """

    def plot_coverage(self, res=1e6):

        n_windows = int(np.ceil(self.last_position / res))
        windows = np.linspace(
            (0, res), ((n_windows - 1) * res, n_windows * res), n_windows,
            dtype=np.int64
        )
        coverages = self.get_window_coverages(windows)
        scale = res / 1e6
        fig = plt.figure(figsize=(10, 2))
        sub = fig.add_subplot(111)
        i = 0
        for i, coverage in enumerate(coverages):
            if coverage == 0:
                color = "yellow"
            else:
                color = cm.binary(coverage)
            sub.barh(y=0.5, height=1, left=i * scale, width=(i + 1) * scale,
                     color=color)
        scaled_end = (i + 1) * scale
        sub.set_yticks(np.arange(0, scaled_end, 5))
        sub.set_xlim(0, scaled_end)
        sub.set_xlabel("Mb")
        sub.set_ylim(0, 1)
        fig.tight_layout()
        fig.show()

    def plot_length_distribution(self, bins=50, y_max=None):
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

    """
    Writing to file
    """

    def write_bed(self, file_name):
        """
        Save the regions array as a .bed file.
        """
        with open(file_name, 'w') as file:
            for i in np.arange(self.n_regions):
                file.write(
                    "\t".join(
                        [
                            str(self.chroms[i]),
                            str(self.regions[i, 0]),
                            str(self.regions[i, 1])
                        ]
                    ) + "\n"
                )
        print(f".bed file for chr {self.chrom} with {self.n_positions} "
              f"positions written at {file_name}")


def intersect_beds(*beds):
    """
    Return the intersection of regions in two or more Bed instances
    """
    n_beds = len(beds)
    max_length = max([bed.last_position for bed in beds])
    indicators = [bed.get_sized_indicator(length=max_length) for bed in beds]
    n_overlaps = np.sum(indicators, axis=0)
    overlaps = np.where(n_overlaps == n_beds)[0]
    chroms = beds[0].chroms
    intersect = Bed.from_positions_0(overlaps, chroms)
    return intersect


def union_beds(*beds):
    """
    Return the union of the regions in two or more Bed instances
    """
    max_length = max([bed.last_position for bed in beds])
    indicators = [bed.get_sized_indicator(length=max_length) for bed in beds]
    n_overlaps = np.sum(indicators, axis=0)
    non_zeros = np.nonzero(n_overlaps)[0]
    union = Bed.from_positions_0(non_zeros, beds[0].chroms)
    return union


def subtract_bed(bed, exclude):
    """
    Get a Bed instance holding regions that are present in bed and not present
    in exclude
    """
    length = max(bed.last_position, exclude.last_position)
    indicator = bed.get_sized_indicator(length=length)
    excl_indicator = exclude.get_sized_indicator(length=length)
    indicator -= excl_indicator
    positions = np.where(indicator == 1)[0]
    subset = Bed.from_positions_0(positions, bed.chroms)
    return subset


def exclude_low_coverage(bed, bin_size, min_coverage):

    n_bins = int(bed.last_position // bin_size) + 1
    bins = np.arange(0, (n_bins + 1) * bin_size, bin_size)
    positions = bed.positions_0
    pos_counts, dump = np.histogram(positions, bins=bins)
    coverage = pos_counts / bin_size
    indicator = np.zeros(bed.last_position, dtype=np.uint8)
    for i in np.arange(n_bins):
        if coverage[i] >= min_coverage:
            indicator[bins[i]:bins[i + 1]] = 1
    coverage_mask = Bed.from_boolean_mask_0(indicator > 0, bed.chroms)
    out = intersect_beds(bed, coverage_mask)
    return out
