
"""
A class for representing .vcf files
"""

import numpy as np
from util import vcf_util
from util import map_util
from util import bed_util


data_path = "/home/nick/Projects/archaic/data"


class USampleSet:

    """
    Class for loading arrays of unphased genotypes from .vcf files containing
    only variant sites.
    """

    def __init__(self, chrom, genotypes, variant_sites, positions=None,
                 position_map=None):

        # required elements
        self.chrom = int(chrom)
        self.sample_ids = list(genotypes.keys())
        self.n_samples = len(self.sample_ids)
        self.genotypes = genotypes
        self.variant_sites = variant_sites
        self.n_variant_sites = len(self.variant_sites)
        # optional elements
        self.positions = positions
        if type(positions) == np.ndarray:
            self.variant_idx = np.searchsorted(positions, variant_sites)
            self.n_positions = len(self.positions)
        else:
            self.variant_idx = None
            self.n_positions = None
        if type(position_map) == np.ndarray:
            self.position_map = position_map
            self.variant_site_map = self.position_map[self.variant_idx]
        else:
            self.position_map = None
            self.variant_site_map = None

    """
    Instantiation
    """

    @classmethod
    def read(cls, vcf_file_name, bed_file_name, map_file_name):
        """
        If there are sites in the vcf file which are not covered in the bed
        file, they will not be loaded. Conversely, bed file regions which
        weren't represented in the vcf will be loaded and treated as ref

        The class works this way so that vcf files can contain variants only.
        be careful!
        """
        variant_sites, genotypes = vcf_util.read(vcf_file_name)
        bed = bed_util.Bed.read_bed(bed_file_name)
        genetic_map = map_util.GeneticMap.read_txt(map_file_name)
        positions = bed.positions_1
        position_map = genetic_map.approximate_map_values(positions)
        variant_mask = cls.get_variant_idx(bed, variant_sites)
        variant_sites = variant_sites[variant_mask]
        for sample_id in genotypes:
            genotypes[sample_id] = genotypes[sample_id][variant_mask]
        chrom = bed.chrom
        return cls(chrom, genotypes, variant_sites, positions=positions,
                   position_map=position_map)

    @staticmethod
    def get_variant_idx(bed, sites):
        """
        Get an index in variant_sites to select sites represented in a bed
        regions array.

        :param bed:
        :param sites: 1-indexed vector of site positions
        """
        edges = bed.flat_regions
        # sites in bed regions have odd region indices, sites outside have even
        edge_idx = np.searchsorted(edges, sites)
        indicator = np.nonzero(edge_idx % 2)[0]
        return indicator

    @classmethod
    def read_vcf(cls, vcf_file_name):
        # just read a .vcf file
        variant_sites, genotypes = vcf_util.read(vcf_file_name)
        chrom = vcf_util.read_chrom(vcf_file_name)
        return cls(chrom, genotypes, variant_sites)

    @classmethod
    def read_chr(cls, chrom):
        """
        Load from pre-specified directories
        """
        vcf_path = f"{data_path}/chrs/chr{chrom}.vcf.gz"
        bed_path = f"{data_path}/masks//chr{chrom}.bed"
        map_path = f"{data_path}/maps/chr{chrom}_map.txt"
        return cls.read(vcf_path, bed_path, map_path)

    @classmethod
    def read_npz(cls, chrom):
        """
        Load from a .npz file
        """
        file_name = f"{data_path}/npz/chr{chrom}.npz"
        npz_file = np.load(file_name)
        positions = npz_file["positions"]
        variant_sites = npz_file["variant_sites"]
        position_map = npz_file["position_map"]
        genotypes = {}
        for key in npz_file:
            if "genotype" in key:
                sample_id = key.split(':')[1]
                genotypes[sample_id] = npz_file[key]
        return cls(chrom, genotypes, variant_sites, positions=positions,
                   position_map=position_map)

    """
    Properties
    """

    @property
    def first_position(self):
        return self.positions[0]

    @property
    def last_position(self):
        return self.positions[-1]

    @property
    def big_window(self):

        return self.first_position, self.last_position + 1

    """
    Accessing variants for a single sample
    """

    def genotype(self, sample_id):
        """
        Return the genotype array belonging to sample
        """
        return self.genotypes[sample_id]

    def alt_counts(self, sample_id):
        """
        Return a vector of alternate allele counts, mapped to
        self.variant_sites
        """
        return np.sum(self.genotypes[sample_id] > 0, axis=1)

    def sample_variant_idx(self, sample_id):
        """
        Return a vector that indexes this sample's variant sites in
        self.variant_sites
        """
        return np.nonzero(self.alt_counts(sample_id) > 0)[0]

    def het_idx(self, sample_id):
        """
        Return a vector that indexes this sample's variant sites in
        self.variant_sites
        """
        genotypes = self.genotypes[sample_id]
        return np.nonzero(genotypes[:, 0] != genotypes[:, 1])[0]

    def het_indicator(self, sample_id, window=None):
        """
        Return an indicator vector on self.variant_positions for heterozygosity
        for a given sample_id
        """
        genotypes = self.genotypes[sample_id]
        if window:
            window_idx = self.idx_variant_window(window)
            genotypes = genotypes[window_idx]
        else:
            pass
        return genotypes[:, 0] != genotypes[:, 1]

    def count_het_sites(self, sample_id, bounds=None):
        """
        Return the number of heterozygous sites for a given sample_id
        """
        het_indicator = self.het_indicator(sample_id)
        if bounds:
            bound_slice = self.get_variant_slice(bounds)
            het_indicator = het_indicator[bound_slice]
        else:
            pass
        return het_indicator.sum()

    """
    Computing statistics for multiple samples 
    """

    def all_alt_counts(self, window=None):

        if window:
            idx = self.idx_variant_window(window)
            alt_counts = np.array(
                [self.alt_counts(sample_id)[idx]
                 for sample_id in self.sample_ids]
            )
        else:
            alt_counts = np.array(
                [self.alt_counts(sample_id) for sample_id in self.sample_ids]
            )
        allele_counts = np.sum(alt_counts, 0)
        return allele_counts

    def get_alt_afs(self, window=None):
        """


        :param window: If provided, restruct
        """
        allele_counts = self.all_alt_counts(window=window)
        possible_counts = np.arange(1, 2 * self.n_samples)
        afs = np.array(
            [np.count_nonzero(allele_counts == i) for i in possible_counts]
        )
        return afs

    def get_folded_afs(self, window=None):

        afs = self.get_alt_afs(window=window)
        n = self.n_samples
        folded_afs = np.zeros(n)
        folded_afs += afs[:n]
        folded_afs[:-1] += afs[:n - 1:-1]
        return folded_afs

    def site_diff_probs(self, sample_id_x, sample_id_y):
        genotypes_x = self.genotype(sample_id_x)
        genotypes_y = self.genotype(sample_id_y)
        probs = (
                (genotypes_x[:, 0][:, np.newaxis] != genotypes_y).sum(1)
                + (genotypes_x[:, 1][:, np.newaxis] != genotypes_y).sum(1)
        )
        probs = probs / 4
        return probs

    """
    Accessing variants for multiple samples
    """

    def multi_sample_variant_idx(self, *sample_ids):
        """
        Return a vector that indexes the union of variant sites in an
        arbitrary number of samples given by *sample_ids

        :param sample_ids:
        """
        sample_variant_set = set(
            np.concatenate(
                (
                    [self.sample_variant_idx(sample_id)
                     for sample_id in sample_ids]
                )
            )
        )
        idx = np.sort(np.array(list(sample_variant_set), dtype=np.int64))
        return idx

    """
    Indexing variants
    """

    def idx_variant_window(self, window):
        """
        Return an index on self.variant_sites for sites inside a window

        :param window:
        """
        lower, upper = window
        window_idx = np.where(
            (self.variant_sites >= lower) & (self.variant_sites < upper)
        )[0]
        return window_idx

    def slice_variant_window(self, window):
        """
        Return a slice on self.variant_sites that accesses sites in a window

        :param window:
        """
        lower, upper = window
        lower_idx = np.searchsorted(self.variant_sites, lower)
        upper_idx = np.searchsorted(self.variant_sites, upper)
        slc = slice(lower_idx, upper_idx)
        return slc

    """
    Accessing positions
    """

    def position_count(self, window):
        """
        Return the number of positions that fall in a genomic window, treating
        the upper window bound as noninclusive

        :param window:
        """
        count = np.count_nonzero(
            (self.positions >= window[0]) & (self.positions < window[1])
        )
        return count

    def variant_sites(self, sample_id):
        """
        Return a vector of sites where a sample has variants
        """
        return self.variant_sites[self.sample_variant_idx(sample_id)]

    def het_sites(self, sample_id):
        """
        Return a vector of sites where a sample is heterozygous
        """
        return self.variant_sites[self.het_idx(sample_id)]

    """
    Windows
    """

    def get_slice(self, bounds):
        """
        Return a slice object that accesses the positions within given bounds
        (the upper bound is noninclusive)
        """
        out = slice(
            np.searchsorted(self.positions, bounds[0]),
            np.searchsorted(self.positions, bounds[1])
        )
        return out

    def get_variant_slice(self, bounds):
        """
        Return a slice object that accesses variant positions within given
        bounds (the upper bound is noninclusive)
        """
        out = slice(
            np.searchsorted(self.variant_sites, bounds[0]),
            np.searchsorted(self.variant_sites, bounds[1])
        )
        return out

    """
    Accessing the map
    """

    def het_map(self, sample_id):
        """
        Return a vector of map values at heterozygous sites for a given
        sample_id
        """
        return self.variant_site_map[self.het_idx(sample_id)]

    """
    Writing to file
    """

    def write_npz(self, file_name):
        """
        Write all the vectors in the instance into a .npz file
        """
        kwargs = {
            "positions": self.positions,
            "variant_sites": self.variant_sites,
            "position_map": self.position_map,
            "chrom": self.chrom
        }
        for sample_id in self.sample_ids:
            kwargs[f"genotype:{sample_id}"] = self.genotypes[sample_id]
        np.savez(file_name, **kwargs)
