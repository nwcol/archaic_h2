import numpy as np

import os

import sys

import gzip

sys.path.append("/home/nick/Projects/archaic/src")

import archaic.vcf_util as vcf_util

import archaic.map_util as map_util

import archaic.bed_util as bed_util


class UnphasedSampleSet:

    def __init__(self, sample_ids, positions, variant_positions, genotypes,
                 genetic_map):

        self.sample_ids = sample_ids
        self.genotypes = genotypes
        self.positions = positions
        self.variant_positions = variant_positions
        self.variant_idx = np.searchsorted(positions, variant_positions)
        self.map_values = genetic_map.approximate_map_values(positions)
        self.variant_map_values = self.map_values[self.variant_idx]

    @classmethod
    def from_one_vcf(cls, vcf_path, bed_path, map_path):

        sample_ids = [x.decode() for x in vcf_util.read_sample_ids(vcf_path)]
        genotypes = {sample_id: load_genotypes(vcf_path, sample_id)
                     for sample_id in sample_ids}
        variant_positions = vcf_util.read_positions(vcf_path)
        genetic_map = map_util.GeneticMap.load_txt(map_path)
        positions = bed_util.Bed.load_bed(bed_path).get_positions_1()
        return cls(sample_ids, positions, variant_positions, genotypes,
                   genetic_map)


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
        return cls.from_one_vcf(vcf_path, bed_path, map_path)

    @property
    def n_samples(self):
        return len(self.sample_ids)

    @property
    def n_positions(self):
        return len(self.positions)

    @property
    def n_variants(self):
        return len(self.variant_positions)

    def alt_allele_counts(self, sample_id):
        alt_counts = np.sum(self.genotypes[sample_id], axis=1)
        return alt_counts

    def alt_allele_freqs(self, sample_id):
        return (self.alt_allele_counts(sample_id) / 2).astype(np.float64)

    def heterozygous_loci_index(self, sample_id):
        """
        Return a vector which indexes heterozygous loci in the genotypes
        array

        :param sample_id:
        :return:
        """
        genotypes = self.genotypes[sample_id]
        return np.nonzero(genotypes[:, 0] != genotypes[:, 1])[0]


    def heterozygous_loci(self, sample_id):
        """
        Return a vector which indexes heterozygous loci in the positions array

        :param sample_id:
        :return:
        """
        idx_in_variants = self.heterozygous_loci_index(sample_id)
        idx_in_positions = self.variant_idx[idx_in_variants]
        return self.positions[idx_in_positions]


def count_alternate_alleles(genotype):

    alleles = genotype.split(b'/')
    count = 0
    for allele in alleles:
        if allele != b'0':
            count += 1
    return count


def decode_genotype(genotype_bytes):

    alleles = np.array(
        [allele.decode() for allele in genotype_bytes.split(b'/')],
        dtype=np.uint8)
    return alleles


def load_genotypes(path, sample_id):

    sample_id = sample_id.encode()
    samples = vcf_util.read_sample_ids(path)
    column = samples[sample_id]
    genotypes = []
    with gzip.open(path, 'r') as file:
        for line in file:
            if b'#' not in line:
                genotype_bytes = vcf_util.parse_genotype(line, column)
                genotypes.append(decode_genotype(genotype_bytes))
    return np.array(genotypes)




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
        self.positions = bed.get_positions_1()
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
        genetic_map = map_util.GeneticMap.load_txt(map_path)
        bed = bed_util.Bed.load_bed(bed_path)
        return cls(samples, alt_positions, genetic_map, bed)

    @classmethod
    def multi_file(cls, path):
        """
        Load multiple .vcfs, each containing a single sample. Intended for
        simulated .vcfs

        :return:
        """
        bed_path = None
        map_path = None
        files = os.listdir(path)
        vcfs = [file for file in files if ".vcf.gz" in file]
        for file in files:
            if ".bed" in file:
                bed_path = path + file
            elif "genetic_map" in file:
                map_path = path + file
        samples = dict()
        positions = []
        for file_name in vcfs:
            sample_id = file_name.strip(".vcf.gz")
            sample_dict, pos = vcf_util.read_samples(path + file_name)
            positions.append(pos)
            samples[sample_id] = sample_dict[b"tsk_0"]
        alt_positions = np.sort(np.concatenate(positions))
        genetic_map = map_util.GeneticMap.load_txt(map_path)
        bed = bed_util.Bed.load_bed(bed_path)
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


# samples = UnphasedSamples.dir(
# "/home/nick/Projects/archaic/data/chromosomes/merged/chr22/")
