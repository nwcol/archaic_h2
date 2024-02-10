
#

import numpy as np
from util import vcf_util
from util import map_util
from util import bed_util


data_path = "/home/nick/Projects/archaic/data"


class SampleSet:

    a = 0


class UnphasedSampleSet:

    def __init__(self, genotypes, positions, variant_sites, map_values, chrom):

        # structural elements
        self.sample_ids = list(genotypes.keys())
        self.genotypes = genotypes
        self.positions = positions
        self.variant_sites = variant_sites
        self.variant_idx = np.searchsorted(positions, variant_sites)
        self.position_map = map_values
        self.variant_site_map = self.position_map[self.variant_idx]
        # properties
        self.chrom = chrom
        self.n_samples = len(self.sample_ids)
        self.n_positions = len(self.positions)
        self.first_position = self.positions[0]
        self.last_position = self.positions[-1]
        self.n_variant_sites = len(self.variant_sites)
        self.big_window = (self.first_position, self.last_position + 1)

    @classmethod
    def read(cls, vcf_path, bed_path, map_path):
        """

        :param vcf_path:
        :param bed_path:
        :param map_path:
        :return:
        """
        variant_sites, genotypes = vcf_util.read(vcf_path)
        genetic_map = map_util.GeneticMap.read_txt(map_path)
        bed = bed_util.Bed.read_bed(bed_path)
        positions = bed.get_1_idx_positions()
        chrom = bed.chrom
        map_values = genetic_map.approximate_map_values(positions)
        return cls(genotypes, positions, variant_sites, map_values, chrom)

    @classmethod
    def read_chr(cls, chrom):
        # make the paths configurable!
        vcf_path = f"{data_path}/chrs/chr{chrom}/chr{chrom}_intersect.vcf.gz"
        bed_path = f"{data_path}/masks/chr{chrom}/chr{chrom}_intersect.bed"
        map_path = f"{data_path}/maps/chr{chrom}_genetic_map.txt"
        return cls.read(vcf_path, bed_path, map_path)

    # Accessing variants for one sample

    def get_sample_alt_counts(self, sample_id):
        """
        Return a vector of alternate allele counts, mapped to
        self.variant_sites
        """
        return np.sum(self.genotypes[sample_id] > 0, axis=1)

    def get_sample_variant_idx(self, sample_id):
        """
        Return a vector that indexes this sample's variant sites in
        self.variant_sites
        """
        return np.nonzero(self.get_sample_alt_counts(sample_id) > 0)[0]

    def get_sample_het_idx(self, sample_id):
        """
        Return a vector that indexes this sample's variant sites in
        self.variant_sites
        """
        genotypes = self.genotypes[sample_id]
        return np.nonzero(genotypes[:, 0] != genotypes[:, 1])[0]

    def get_sample_het_indicator(self, sample_id, window=None):
        """
        Return an indicator vector on self.variant_positions for heterozygosity
        for a given sample_id
        """
        genotypes = self.genotypes[sample_id]
        if window:
            window_idx = self.get_window_variant_idx(window)
            genotypes = genotypes[window_idx]
        else:
            pass
        return genotypes[:, 0] != genotypes[:, 1]

    def get_n_sample_variants(self, sample_id):
        """
        Return the number of sites that vary from the reference for a sample
        """
        return len(self.get_sample_variant_idx(sample_id))

    def get_n_sample_het_sites(self, sample_id, window=None):
        """
        Return the number of heterozygous sites for a given sample_id
        """
        het_indicator = self.get_sample_het_indicator(sample_id)
        if window:
            window_idx = self.get_window_variant_idx(window)
            het_indicator = het_indicator[window_idx]
        else:
            pass
        return np.sum(het_indicator)

    # Accessing variants for multiple samples

    def get_multi_sample_variant_idx(self, *sample_ids):
        """
        Return a vector that indexes the union of variant sites in an
        arbitrary number of samples given by *sample_ids

        :param sample_ids:
        """
        sample_variant_set = set(
            np.concatenate(
                (
                    [self.get_sample_variant_idx(sample_id)
                     for sample_id in sample_ids]
                )
            )
        )
        return np.sort(np.array(list(sample_variant_set), dtype=np.int64))

    # Indexing variants

    def get_window_variant_idx(self, window):
        """
        Return an index on self.variant_sites for sites inside a window

        :param window:
        """
        window_idx = np.where(
            (self.variant_sites >= window[0])
            & (self.variant_sites < window[1])
        )[0]
        return window_idx

    # Accessing positions

    def get_window_position_count(self, window):
        """
        Return the number of positions that fall in a genomic window, treating
        the upper window bound as noninclusive

        :param window:
        """
        count = np.count_nonzero(
            (self.positions >= window[0]) & (self.positions < window[1])
        )
        return count

    def get_sample_variant_sites(self, sample_id):
        """
        Return a vector of sites where a sample has variants

        :param sample_id:
        """
        return self.variant_sites[self.get_sample_variant_idx(sample_id)]

    def get_sample_het_sites(self, sample_id):
        """
        Return a vector of sites where a sample is heterozygous

        :param sample_id:
        """
        return self.variant_sites[self.get_sample_het_idx(sample_id)]

    # Accessing map

    def get_sample_variant_map(self, sample_id):
        """
        Return a vector of map values at variant sites for a given sample_id
        """
        return self.variant_site_map[self.get_sample_variant_idx(sample_id)]


    def get_sample_het_map(self, sample_id):
        """
        Return a vector of map values at heterozygous sites for a given
        sample_id
        """
        return self.variant_site_map[self.get_sample_het_idx(sample_id)]


class PhasedSampleSet:

    def __init__(self):
        # there probably isn't a need for this
        pass
