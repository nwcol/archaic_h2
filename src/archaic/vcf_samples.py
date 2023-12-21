import numpy as np

import os

if __name__ == "__main__":
    import vcf_util

    import map_util

    import bed_util

else:
    from src.archaic import vcf_util

    from src.archaic import map_util

    from src.archaic import bed_util


class Samples:

    def __init__(self, samples, alt_positions, genetic_map, bed):
        """

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
        self.map_values = genetic_map.approximate_map_values(self.positions)
        self.alt_index = np.searchsorted(self.positions, self.alt_positions)
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
    def multi_file(cls, vcf_dir, bed_path, map_path):
        """
        Load multiple .vcfs, each containing a single sample. Intended for
        simulated .vcfs

        :param vcf_dir:
        :param bed_path:
        :param map_path:
        :return:
        """
        files = [file for file in os.listdir(vcf_dir) if ".vcf.gz" in file]
        samples = dict()
        positions = dict()
        for file_name in files:
            sample_id = file_name.strip(".vcf.gz")
            sample_dict, positions = vcf_util.read_samples(vcf_dir + file_name)

        alt_positions = 0

        #### FINISH LATER

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


samples = Samples.dir("c:/archaic/data/chromosomes/merged/chr22/")
