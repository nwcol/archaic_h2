
#

import numpy as np
import os
from util import vcf_util
from util import map_util
from util import bed_util


data_path = "/home/nick/Projects/archaic/data"


class SampleSet:

    a = 0


class UnphasedSampleSet:

    def __init__(self, genotypes, positions, variant_positions, map_values,
                 chrom):

        self.sample_ids = list(genotypes.keys())
        self.chrom = chrom
        self.genotypes = genotypes
        self.positions = positions
        self.variant_positions = variant_positions
        self.variant_idx = np.searchsorted(positions, variant_positions)
        self.map_values = map_values
        self.variant_map_values = self.map_values[self.variant_idx]

    @classmethod
    def read(cls, vcf_path, bed_path, map_path):
        """

        :param vcf_path:
        :param bed_path:
        :param map_path:
        :return:
        """
        variant_positions, genotypes = vcf_util.read(vcf_path)
        genetic_map = map_util.GeneticMap.read_txt(map_path)
        bed = bed_util.Bed.read_bed(bed_path)
        positions = bed.get_1_idx_positions()
        chrom = bed.chrom
        map_values = genetic_map.approximate_map_values(positions)
        return cls(genotypes, positions, variant_positions, map_values, chrom)

    @classmethod
    def get_chr(cls, chrom):
        # make the paths configurable!
        vcf_path = f"{data_path}/chrs/chr{chrom}/chr{chrom}_intersect.vcf.gz"
        bed_path = f"{data_path}/masks/chr{chrom}/chr{chrom}_intersect.bed"
        map_path = f"{data_path}/maps/chr{chrom}_genetic_map.txt"
        return cls.read(vcf_path, bed_path, map_path)

    def slice(self, start_pos, end_pos):
        """
        Return a positional subset of the UnphasedSampleSet.

        start_pos and end_pos are 1-indexed and refer to positions along the 
        chromosome. end_pos is non-inclusive.
        """
        pos = self.positions
        pos_idx = np.where((pos >= start_pos) & (pos < end_pos))[0]
        positions = self.positions[pos_idx]

        var_pos = self.variant_positions
        variant_idx = np.where((var_pos >= start_pos) & (var_pos < end_pos))[0]
        variant_positions = var_pos[variant_idx]
        genotypes = {sample_id: self.genotypes[sample_id][variant_idx] for
                     sample_id in self.sample_ids}

        map_values = self.map_values[pos_idx]
        return UnphasedSampleSet(genotypes, positions, variant_positions,
                                 map_values)

    @property
    def n_samples(self):
        """
        Return the number of samples represented in the instance
        """
        return len(self.sample_ids)

    @property
    def n_positions(self):
        """
        Return the number of positions represented in the instance
        """
        return len(self.positions)

    def interval_n_positions(self, start, end):
        """
        Get the number of positions on an interval
        """
        n = np.searchsorted(self.positions, end) \
            - np.searchsorted(self.positions, start)
        return n

    @property
    def n_variants(self):
        """
        Return the number of sites variant in one or more samples
        """
        return len(self.variant_positions)

    @property
    def min_position(self):
        """
        Return the lowest-index position represented in the instance
        """
        return self.positions[0]

    @property
    def max_position(self):
        """
        Return the highest-index position represented in the instance
        """
        return self.positions[-1]

    def get_short_variant_idx(self, sample_id):
        """
        Return a vector that indexes this sample's variant sites in
        self.variant_positions
        """
        return np.nonzero(self.get_short_alt_counts(sample_id) > 0)[0]

    def get_short_alt_counts(self, sample_id):
        """
        Return a vector of alternate allele counts, mapped to
        self.variant_positions
        """
        alt_counts = np.sum(self.genotypes[sample_id] > 0, axis=1)
        return alt_counts

    def get_variant_idx(self, sample_id):
        """
        Return a vector that indexes this sample's variant sites in
        self.positions
        """
        return np.nonzero(self.get_alt_counts(sample_id) > 0)[0]

    def get_alt_counts(self, sample_id):
        """
        Return a vector of alternate allele counts, mapped to self.positions
        """
        counts = np.sum(self.genotypes[sample_id] > 0, axis=1)
        alt_counts = np.zeros(self.n_positions, dtype=np.uint8)
        alt_counts[self.variant_idx] = counts
        return alt_counts

    def get_het_indicator(self, sample_id):
        """
        Return a indicator vector on self.variant_positions for heterozygosity
        for a given sample_id
        """
        het_indicator = np.zeros(self.n_positions, dtype=np.int8)
        het_indicator[self.get_het_idx(sample_id)] = 1
        return het_indicator

    def get_het_idx(self, sample_id):
        """
        Return the index of heterozygous positions in self.positions for a
        given sample_id
        """
        genotypes = self.genotypes[sample_id]
        return self.variant_idx[genotypes[:, 0] != genotypes[:, 1]]

    def get_het_positions(self, sample_id):
        """
        Return the vector of heterozygous positions for a given sample_id
        """
        return self.positions[self.get_het_idx(sample_id)]

    def get_het_map(self, sample_id):
        """
        Return a vector of map values at heterozygous sites for a given
        sample_id
        """
        return self.map_values[self.get_het_idx(sample_id)]

    def n_het(self, sample_id):
        """
        Return the number of heterozygous sites for a given sample_id
        """
        return np.sum(self.get_het_indicator(sample_id))

    def index_position(self, position):
        """
        Return the index of a given position in self.positions
        If the position does not exist in that array, return the index which
        accesses the next position above it
        """
        return np.searchsorted(self.positions, position)

    def index_het_position(self, sample_id, position):

        return np.searchsorted(self.get_het_positions(sample_id), position)



class PhasedSampleSet:

    def __init__(self):
        pass
