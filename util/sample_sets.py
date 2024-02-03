
#

import numpy as np
import os
from util import vcf_util
from util import map_util
from util import bed_util


class SampleSet:

    a = 0


class UnphasedSampleSet:

    def __init__(self, genotypes, positions, variant_positions, map_values):
        """

        :param positions:
        :param variant_positions:
        :param genotypes: dictionary
        :param map_values:
        """
        self.sample_ids = list(genotypes.keys())
        self.genotypes = genotypes
        self.positions = positions
        self.variant_positions = variant_positions
        self.variant_idx = np.searchsorted(positions, variant_positions)
        self.map_values = map_values
        self.variant_map_values = self.map_values[self.variant_idx]

    @classmethod
    def from_files(cls, vcf_path, bed_path, map_path):
        
        variant_positions, genotypes = vcf_util.read(vcf_path)
        genetic_map = map_util.GeneticMap.read_txt(map_path)
        positions = bed_util.Bed.read_bed(bed_path).get_1_idx_positions()
        map_values = genetic_map.approximate_map_values(positions)
        return cls(genotypes, positions, variant_positions, map_values)

    @classmethod
    def from_dir(cls, path):

        vcf_path = None
        bed_path = None
        map_path = None
        files = os.listdir(path)
        for file in files:
            if "merged.vcf.gz" in file:
                vcf_path = path + file
            elif ".bed" in file:
                bed_path = path + file
            elif "genetic_map" in file:
                map_path = path + file
        if not vcf_path:
            raise ValueError("no merged .vcf.gz!")
        if not bed_path:
            raise ValueError("no merged .bed!")
        if not map_path:
            raise ValueError("no genetic map!")
        return cls.from_files(vcf_path, bed_path, map_path)

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

    @property
    def n_variants(self):
        """
        Return the number of sites variant in one or more samples
        """
        return len(self.variant_positions)

    @property
    def start(self):
        """
        Return the lowest-index position represented in the instance
        """
        return self.positions[0]

    @property
    def stop(self):
        """
        Return the highest-index position represented in the instance
        """
        return self.positions[-1]

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


sample_set = UnphasedSampleSet.from_dir(
    "/home/nick/Projects/archaic/data/chromosomes/merged/chr22/")


class PhasedSampleSet:

    def __init__(self):
        pass


class UnphasedSamples:

    def __init__(self, samples, alt_positions, genetic_map, bed):
        """
        alt_positions gives the chromosomal positions with variants, and
        alt_index gives the index of these sites in the positions vector.

        :param samples:
        :param alt_positions:
        :param genetic_map:
        :param bed:
        """
        self.samples = samples
        self.sample_ids = list(samples.keys())
        self.n_samples = len(samples)
        self.alt_positions = alt_positions
        self.positions = bed.get_1_idx_positions()
        self.n_positions = len(self.positions)
        self.alt_index = np.searchsorted(self.positions, self.alt_positions)
        self.n_variants = len(self.alt_positions)
        self.map_values = genetic_map.approximate_map_values(self.positions)
        self.alt_map_values = self.map_values[self.alt_index]

    @classmethod
    def one_file(cls, vcf_path, bed_path, map_path):
        """
        Load multiple samples from a single .vcf

        :return:
        """
        encoded_samples, alt_positions = vcf_util.read_samples(vcf_path)
        samples = dict()
        for sample_id in encoded_samples:
            samples[sample_id.decode()] = encoded_samples[sample_id]
        genetic_map = map_util.GeneticMap.read_txt(map_path)
        bed = bed_util.Bed.read_bed(bed_path)
        return cls(samples, alt_positions, genetic_map, bed)

    @classmethod
    def dir(cls, path):
        vcf_path = None
        bed_path = None
        map_path = None
        files = os.listdir(path)
        for file in files:
            if "merged.vcf.gz" in file:
                vcf_path = path + file
            elif ".bed" in file:
                bed_path = path + file
            elif "genetic_map" in file:
                map_path = path + file
        if not vcf_path:
            raise ValueError("no merged .vcf.gz!")
        if not bed_path:
            raise ValueError("no merged .bed!")
        if not map_path:
            raise ValueError("no genetic map!")
        return cls.one_file(vcf_path, bed_path, map_path)

    def alts(self, sample_id):
        """
        For a given sample, return a vector of alternate alle counts for every
        position in self.positions

        :param sample_id:
        :return:
        """
        genotypes = np.zeros(self.n_positions, dtype=np.uint8)
        genotypes[self.alt_index] = self.samples[sample_id]
        return genotypes

    def het_indicator(self, sample_id):
        """
        Return an indicator vector for heterozygosity in one sample.

        :param sample_id:
        :return:
        """
        indicator = np.zeros(self.n_positions, dtype=np.uint8)
        indicator[self.alts(sample_id) == 1] = 1
        return indicator

    def het_index(self, sample_id):
        """
        Return a vector of integers that index a sample's heterozygous sites
        in the self.positions array

        :param sample_id:
        :return:
        """
        het_index = self.alt_index[self.samples[sample_id] == 1]
        return het_index

    def n_het(self, sample_id):
        """
        Return the number of heterozygous sites for one sample

        :param sample_id:
        :return:
        """
        n_hets = np.sum(self.samples[sample_id] == 1)
        return n_hets


old = UnphasedSamples.dir("/home/nick/Projects/archaic/data/chromosomes/merged/chr22/")