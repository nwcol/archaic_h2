"""

"""


import demes
import numpy as np
import moments


class TwoLocusH:

    def __init__(
        self,
        data,
        r_bins,
        ids,
        has_H=True,
        covariances=None
    ):
        """
        :param data: An array of H2 values with shape (n bins, n samples).
            If H is included the shape is (n bins + 1, n samples). The H row
            should be the last.
        """
        self.data = data
        self.covariances = covariances
        if covariances is not None:
            self.inv_covariances = self.invert_covariances(covariances)
        self.r_bins = r_bins
        self.ids = ids
        self.has_H = has_H

    @classmethod
    def from_npz_file(cls, fname):
        # placeholder. I am going to change the file names in these archives
        file = np.load(fname)
        data = np.vstack([file['H2_mean'], file['H_mean']])
        covs = np.vstack([file['H2_cov'], file['H_cov'][np.newaxis, :, :]])
        r_bins = file['r_bins']
        sample_ids = file['sample_names']
        return cls(data, r_bins, sample_ids, has_H=True, covariances=covs)

    @classmethod
    def from_graph(cls, graph, sample_ids, r_bins, u, r=None):
        #
        stats = moments.LD.LDstats.from_demes(
            graph, sampled_demes=sample_ids, theta=None, r=r, u=u
        )
        n = len(sample_ids)
        exp_H = stats.H()
        _exp_H2 = np.zeros((len(r), n))
        ids = [
            (sample_ids[i], sample_ids[j])
            for i in np.arange(n) for j in np.arange(i, n)
        ]
        for k, (x, y) in enumerate(ids):
            if x == y:
                phased = True
                y = None
            else:
                phased = False
            _exp_H2[:, k] = stats.H2(x, y, phased=phased)
        exp_H2 = cls.approximate_H2(_exp_H2)
        data = np.vstack([exp_H2, exp_H])
        return cls(data, r_bins, ids, has_H=True)

    @classmethod
    def from_graph_file(cls, fname, sample_ids, r):
        #
        return cls.from_graph(demes.load(fname), sample_ids, r)

    def subset(self):
        # subset by sample id

        return 0

    def subset_idx(self, idx):
        #
        mesh_idx = np.ix_(idx, idx)
        covariances = np.array([x[mesh_idx] for x in self.covariances])
        sub = TwoLocusH(
            self.data[:, idx],
            self.r_bins,
            self.sample_ids[:, idx],
            _pair_ids,
            covariances=covariances)
        return sub

    def subset_bins(self, idx):

        return 0

    @staticmethod
    def invert_covariances(covariances):
        #
        return np.array([np.linalg.inv(x) for x in covariances])

    @staticmethod
    def get_pair_arr(ids):
        #
        n = len(ids)
        arr = np.array(
            [[ids[i], ids[j]] for i in range(n) for j in np.arange(i + 1, n)]
        )
        return arr

    @staticmethod
    def approximate_H2(arr):
        # Simpsons method
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


class TwoLocusSpectrum(np.ndarray):

    pass

