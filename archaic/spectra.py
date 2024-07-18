"""
A class for holding H2 statistics
"""


import demes
import numpy as np
import moments
from archaic import utils


class H2Spectrum:

    def __init__(
        self,
        data,
        r_bins,
        ids,
        sample_ids=None,
        has_H=True,
        covs=None,
        r=None
    ):
        """
        :param data: An array of H2 values with shape (n bins, n samples).
            If H is included the shape is (n bins + 1, n samples). The H row
            should be the last.
        """
        self.data = data
        self.n_bins = len(data)
        self.r_bins = r_bins
        if r_bins is not None:
            if r is None:
                self.r = self.get_r(r_bins)
            else:
                self.r = r
        self.ids = np.asanyarray(ids)
        self.n = len(ids)
        if sample_ids is None:
            self.sample_ids = [str(x) for x in list(np.unique(ids))]
        else:
            self.sample_ids = [str(x) for x in sample_ids]
        self.has_H = has_H
        self.covs = covs
        if covs is not None:
            self.inv_covs = self.invert_cos(covs)

    @classmethod
    def from_bootstrap_file(cls, fname, sample_ids=None, graph=None):
        #
        file = np.load(fname)
        data = np.vstack([file['H2_mean'], file['H_mean']])
        covs = np.vstack([file['H2_cov'], file['H_cov'][np.newaxis, :, :]])
        r_bins = file['r_bins']
        ids = file['ids']
        spectrum = cls(data, r_bins, ids, has_H=True, covs=covs)
        if sample_ids is not None:
            spectrum = spectrum.subset(sample_ids)
        elif graph is not None:
            spectrum = spectrum.subset_to_graph(graph)
        return spectrum

    @classmethod
    def from_bootstrap_distribution(cls, fname, i, sample_ids=None):
        # load the ith resampling from a bootstrap file with 'dist' arrays
        file = np.load(fname)
        data = np.vstack([file['H2_dist'][i], file['H_dist'][i]])
        covs = np.vstack([file['H2_cov'], file['H_cov'][np.newaxis, :, :]])
        r_bins = file['r_bins']
        ids = file['ids']
        spectrum = cls(data, r_bins, ids, has_H=True, covs=covs)
        if sample_ids is not None:
            spectrum = spectrum.subset(sample_ids)
        return spectrum

    @classmethod
    def from_file(cls, fname):
        # from a saved class instance
        file = np.load(fname)
        data = file['data']
        r_bins = file['r_bins']
        ids = file['ids']
        sample_ids = file['sample_ids']
        covs = file['covs']
        has_H = file['has_H']
        spectrum = cls(
            data, r_bins, ids, sample_ids=sample_ids, covs=covs, has_H=has_H
        )
        return spectrum

    def write(self, fname):
        # write as a .npz archive
        dic = dict(
            data=self.data,
            r_bins=self.r_bins,
            ids=self.ids,
            sample_ids=self.sample_ids,
            covs=self.covs,
            has_H=int(self.has_H)
        )
        np.savez(fname, **dic)

    @classmethod
    def from_graph(cls, graph, sample_ids, r, u):
        #
        sample_ids = sorted(sample_ids)
        stats = moments.LD.LDstats.from_demes(
            graph, sampled_demes=sample_ids, theta=None, r=r, u=u
        )
        exp_H = stats.H()
        ids = cls.expand_ids(sample_ids)
        _exp_H2 = np.zeros((len(r), len(ids)))
        for k, (x, y) in enumerate(ids):
            if x == y:
                phased = True
                y = None
            else:
                phased = False
            _exp_H2[:, k] = stats.H2(x, y, phased=phased)
        exp_H2 = cls.approximate_H2(_exp_H2)
        data = np.vstack([exp_H2, exp_H])
        r_bins = None
        spectrum = cls(data, r_bins, np.array(ids), has_H=True)
        return spectrum

    @classmethod
    def from_graph_file(cls, fname, sample_ids, r, u):
        #
        return cls.from_graph(demes.load(fname), sample_ids, r, u)

    @classmethod
    def from_dict(cls, dic):
        #
        if 'H' in dic:
            data = np.vstack([dic['H2'], dic['H']])
            has_H = True
        else:
            data = dic['H2']
            has_H = False
        r_bins = dic['r_bins']
        ids = dic['ids']
        return cls(data, r_bins, ids, has_H=has_H, covs=None)

    def subset(self, sample_ids):
        # subset by sample id
        # sub_ids = self.expand_ids(sample_ids)
        idx = np.array([
            i for i in range(self.n)
            if self.ids[i, 0] in sample_ids and self.ids[i, 1] in sample_ids
        ])
        print(idx)
        return self.subset_idx(idx)

    def subset_idx(self, idx):
        # subset by index
        mesh_idx = np.ix_(idx, idx)
        covs = np.array([x[mesh_idx] for x in self.covs])
        sub = H2Spectrum(
            self.data[:, idx],
            self.r_bins,
            self.ids[idx],
            covs=covs,
            has_H=self.has_H
        )
        return sub

    def subset_to_graph(self, graph):
        # keep only those samples whose names match deme names in a demes graph
        deme_names = [deme.name for deme in graph.demes]
        subset_ids = [_id for _id in self.sample_ids if _id in deme_names]
        return self.subset(subset_ids)

    def subset_bins(self, idx):

        return 0

    def remove_H(self):
        # exclude the one-locus H row of the data array
        if self.has_H:
            if self.covs is None:
                covs = None
            else:
                covs = self.covs[:-1]
            sub = H2Spectrum(
                self.data[:-1],
                self.r_bins,
                self.ids,
                covs=covs
            )
        else:
            print(f'H2Spectrum does not contain H!')
            sub = self
        return sub

    @staticmethod
    def invert_cos(covs):
        #
        return np.array([np.linalg.inv(x) for x in covs])

    @staticmethod
    def approximate_H2(arr):
        # uses Simpsons method
        n = len(arr)
        b = (n - 1) // 2
        ret = (
            1 / 6 * arr[np.arange(b) * 2]
            + 4 / 6 * arr[np.arange(b) * 2 + 1]
            + 1 / 6 * arr[np.arange(b) * 2 + 2]
        )
        return ret

    @staticmethod
    def get_r(r_bins):
        # for approximating H2
        n = len(r_bins)
        r = np.zeros(n * 2 - 1)
        r[np.arange(n) * 2] = r_bins
        r[np.arange(n - 1) * 2 + 1] = r_bins[:-1] + np.diff(r_bins) / 2
        return r

    @staticmethod
    def expand_ids(ids):
        #
        n = len(ids)
        return [(ids[i], ids[j]) for i in range(n) for j in np.arange(i, n)]

    @staticmethod
    def get_pair_ids(ids):
        #
        n = len(ids)
        return [(ids[i], ids[j]) for i in range(n) for j in np.arange(i + 1, n)]
