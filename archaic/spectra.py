"""
A class for holding H2 statistics
"""
import demes
import numpy as np
import moments


_default_r = np.logspace(-6, -2, 17)


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

    @classmethod
    def from_bootstrap_file(cls, fname, sample_ids=None, graph=None):
        #
        file = np.load(fname)
        data = np.vstack([file['H2_mean'], file['H_mean']])
        if file['H_cov'].ndim == 2:
            covs = np.vstack([file['H2_cov'], file['H_cov'][np.newaxis, :, :]])
        else:
            covs = np.vstack(
                [file['H2_cov'], file['H_cov'][np.newaxis, np.newaxis, np.newaxis]]
            )
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
        data = np.vstack([file['H2_dist'][i].T, file['H_dist'][i]])
        covs = np.vstack([file['H2_cov'], file['H_cov'][np.newaxis, :, :]])
        r_bins = file['r_bins']
        ids = file['ids']
        spectrum = cls(data, r_bins, ids, has_H=True, covs=covs)
        if sample_ids is not None:
            spectrum = spectrum.subset(sample_ids)
        return spectrum

    @classmethod
    def from_file(cls, fname, sample_ids=None, graph=None):
        # placeholder. covs are just along windows. masks nan as zero
        file = np.load(fname)
        r_bins = file['r_bins']
        ids = file['ids']
        if 'H2_counts' in file:
            per_window_H2 = file['H2_counts'] / file['n_site_pairs'][:, np.newaxis, :]
            if np.any(np.isnan(per_window_H2)):
                print('setting nan to zero in windowed H2 array')
                per_window_H2[np.isnan(per_window_H2)] = 0
            per_window_H = file['H_counts'] / file['n_sites'][:, np.newaxis]
            # unlikely to happen
            if np.any(np.isnan(per_window_H)):
                print('setting nan to zero in windowed H array')
                per_window_H[np.isnan(per_window_H)] = 0

            if per_window_H2.ndim == 3 and per_window_H2.shape[2] > 1:
                covs = np.array(
                    [np.cov(per_window_H2[:, :, i], rowvar=False)
                     for i in range(per_window_H2.shape[2])]
                    + [np.cov(per_window_H, rowvar=False)]
                )
                if covs.ndim != 3:
                    #n_stats = file['H2_counts'].shape[1]
                    #covs = np.zeros((len(r_bins), n_stats, n_stats))
                    covs = None
                    #n = per_window_H.shape[1]
                    #covs = covs.reshape(len(r_bins), n, n)
            else:
                covs = None
            H2 = file['H2_counts'].sum(0) / file['n_site_pairs'].sum(0)
            H = file['H_counts'].sum(0) / file['n_sites'].sum()
            data = np.vstack([H2.T, H[np.newaxis]])
        elif 'H2' in file:
            H2 = file['H2']
            H = file['H']
            data = np.vstack([H2.T, H[np.newaxis]])
            covs = None


        spectrum = cls(
            data, r_bins, ids, covs=covs, has_H=True
        )
        if sample_ids is not None:
            spectrum = spectrum.subset(sample_ids)
        elif graph is not None:
            spectrum = spectrum.subset_to_graph(graph)
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
    def from_graph(cls, graph, sample_ids, r, u, r_bins=None, get_H=True):
        #
        sample_ids = sorted(sample_ids)
        stats = moments.LD.LDstats.from_demes(
            graph, sampled_demes=sample_ids, theta=None, r=r, u=u
        )
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
        if get_H:
            exp_H = stats.H()
            data = np.vstack([exp_H2, exp_H])
        else:
            data = exp_H2
        spectrum = cls(data, r_bins, np.array(ids), has_H=True)
        return spectrum

    @classmethod
    def from_demes(
        cls,
        graph,
        sampled_demes=None,
        sample_times=None,
        r_bins=None,
        u=1.35e-8
    ):
        # will replace from_graph when I get around to refactoring

        if r_bins is None:
            r_bins = np.logspace(-6, -2, 17)
        r = cls.get_r(r_bins)

        if sampled_demes is None:
            sampled_demes = [d.name for d in graph.demes if d.end_time == 0]
        stats = moments.LD.LDstats.from_demes(
            graph, sampled_demes=sampled_demes, theta=None, r=r, u=u
        )
        exp_H = stats.H()
        ids = cls.expand_ids(sampled_demes)
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
        return self.subset_idx(idx)

    def subset_idx(self, idx):
        # subset by index
        mesh_idx = np.ix_(idx, idx)
        if self.covs is not None:
            covs = np.array([x[mesh_idx] for x in self.covs])
        else:
            covs = None
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
                covs=covs,
                sample_ids=self.sample_ids,
                has_H=False
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

    @property
    def arr(self):
        # alias
        return self.data


class H2stats:
    """
    """

    def __init__(
        self,
        H2,
        covs=None,
        sample_ids=None,
        r_bins=None,
        bin_mask=None,
        sample_mask=None
    ):
        #
        self.H2 = H2
        self.covs = covs
        self.sample_ids = sample_ids
        self.r_bins = r_bins

    def from_file(self):

        return None

    def to_file(self):

        return None

    def from_demes(
        cls,
        graph,
        template_data=None,
        sampled_demes=None,
        u=None,
        r_bins=None,
        sample_sizes=None,
        sample_times=None,
        include_H=True
    ):
        # can be instantiated in several ways:
        # (1) from data: H2stats.from_demes(graph, data, u=u)
        #   then the r bins, sample ids etc are the same as the data
        # (2) from r bins, sampled_demes, and optionally sampled_times
        # (3) just from a graph

        if template_data is not None:
            r_bins = template_data.r_bins
            sampled_demes = template_data.sample_ids
            sample_times = template_data.sample_times

        else:
            if r_bins is None:
                r_bins = _default_r

            if u is None:
                # change?
                u = 1.3e-8

            if sampled_demes is None:
                deme_names = [d.name for d in graph.demes]

        r = cls.r_points(r_bins)

        stats = moments.LD.LDstats.from_demes(
            graph,
            sampled_demes=sampled_demes,
            sample_times=sample_times,
            theta=None,
            r=r,
            u=u
        )

        ids = cls.get_ids(sampled_demes)
        raw_H2 = np.zeros((len(r), len(ids)))
        for k, (x, y) in enumerate(ids):
            if x == y:
                phased = True
                y = None
            else:
                phased = False
            raw_H2[:, k] = stats.H2(x, y, phased=phased)
        H2 = cls.simpsons_method(raw_H2)
        if include_H:
            H = stats.H()
            H2 = np.vstack([H2, H])
        ret = cls(H2)
        return ret

    def from_dict(self):

        return None

    def from_vcf_file(self):

        return None

    def subset(self):

        return None

    @property
    def rs(self):

        if self.r_bins is None:
            return None
        return self.r_points(self.r_bins)

    @staticmethod
    def r_points(r_bins):

        n = len(r_bins)
        r = np.zeros(n * 2 - 1)
        r[np.arange(n) * 2] = r_bins
        r[np.arange(n - 1) * 2 + 1] = r_bins[:-1] + np.diff(r_bins) / 2
        return r

    @staticmethod
    def simpsons_method(arr):

        n = len(arr)
        b = (n - 1) // 2
        ret = (
            1 / 6 * arr[np.arange(b) * 2]
            + 4 / 6 * arr[np.arange(b) * 2 + 1]
            + 1 / 6 * arr[np.arange(b) * 2 + 2]
        )
        return ret






