"""
A class for representing H2 statistics.
"""
import demes
import numpy as np
import numpy.ma as ma
import moments
import pickle


def load_data(in_file, **kwargs):
    """
    Load H2 statistics from one or more dictionaries in a pickled file.
    """
    reps = {}
    with open(in_file, 'rb') as fin:
        dic = pickle.load(fin)
        for rep in dic:
            stats = H2stats.from_dict(dic[rep], **kwargs)
            reps[rep] = stats
    return reps


_default_bins = np.logspace(-6, -2, 17)


class H2stats:
    """
    Holds one and two-locus heterozygosity (H, H2) statistics.

    The `stats` array is 2d, with 0th dimension equal to the number of 
    recombination-distance bins plus one- the last row is reserved for the
    H statistics. The 1st dimension equals the number of samples plus sample
    pairs. Attributes `bins` and `pop_ids` are required. 

    For empirical data obtained by averaging or bootstrapping, `covs` is a 3d 
    array of variance-covariance matrices, whose 0th axis corresponds to the
    recombination bins.
    """

    def __init__(
        self,
        stats, 
        bins=None, 
        pop_ids=None, 
        covs=None, 
        **kwargs
    ):  
        """
        """
        if covs is not None: assert len(stats) == len(covs)
        assert bins is not None
        assert pop_ids is not None
        self.stats = stats
        self.bins = bins
        self.pop_ids = pop_ids
        self.covs = covs

    @classmethod
    def from_dict(cls, dic, to_pop_ids=None, graph=None):
        """
        
        """
        pop_ids = dic['pop_ids']
        bins = dic['bins']
        stats = dic['means']
        covs = dic['covs'] if 'covs' in dic else None
        ret = cls(stats, bins=bins, pop_ids=pop_ids, covs=covs)

        if to_pop_ids is not None:
            ret = ret.subset(to_pop_ids=to_pop_ids)

        elif graph is not None:
            if isinstance(graph, str): 
                graph = demes.load(graph)
            deme_names = [d.name for d in graph.demes]
            overlaps = [pop for pop in pop_ids if pop in deme_names]
            ret = ret.subset(to_pop_ids=overlaps)

        return ret

    @classmethod
    def from_file(cls, fname, to_pop_ids=None, graph=None):
        """
        Loads H2 statistics from the first dictionary in a pickled file.
        """
        with open(fname, 'rb') as fin:
            dics = pickle.load(fin)
            dic = dics[next(iter(dics))]

        return cls.from_dict(dic, to_pop_ids=to_pop_ids, graph=graph)

    @classmethod
    def from_demes(
        cls,
        graph,
        u=None,
        template=None,
        bins=None,
        sampled_demes=None,
        sample_times=None
    ):  
        """
        Instantiate by computing expected statistics from a demes graph using
        moments.LD. The `template` argument matches the bins and pop_ids of 
        the new instance to an existing H2stats instance. 
        """
        if u is None: raise ValueError('argument `u` is required')
        if isinstance(graph, str):
            graph = demes.load(graph)
        # parse arguments
        if template is not None:
            bins = template.bins
            sampled_demes = template.pop_ids
        else:
            bins = bins if bins is not None else _default_bins
            if sampled_demes is None:
                sampled_demes = [
                    d.name for d in graph.demes if d.end_time == 0
                ]
            else:
                for d in sampled_demes:
                    graph_demes = [d.name for d in graph.demes]
                    for d in sampled_demes:
                        if d not in graph_demes: 
                            raise ValueError(f'deme {d} is not present in graph!')
        if sample_times is None:
            _times = {d.name: d.end_time for d in graph.demes}
            sample_times = [_times[pop] for pop in sampled_demes]
        else:
            assert len(sample_times) == len(sampled_demes)

        r_points = get_r_points(bins)
        ld_stats = moments.LD.LDstats.from_demes(
            graph,
            sampled_demes=sampled_demes,
            sample_times=sample_times,
            theta=None,
            r=r_points,
            u=u
        )
        num_demes = len(sampled_demes)
        num_stats = int(num_demes * (num_demes - 1) / 2) + num_demes
        point_H2 = np.zeros((len(r_points), num_stats))
        k = 0
        for i, x in enumerate(sampled_demes):
            for y in sampled_demes[i:]:
                if x == y:
                    phased = True
                    y = None
                else:
                    phased = False
                point_H2[:, k] = ld_stats.H2(x, y, phased=phased)
                k += 1

        H2 = interpolate_quadratic(point_H2)
        stats = np.vstack((H2, ld_stats.H()))

        return cls(stats, bins=bins, pop_ids=sampled_demes, covs=None)
    
    @classmethod
    def from_graph(cls, *args, **kwargs):
        """
        Equivalent to H2stats.from_demes().
        """
        return cls.from_demes(*args, **kwargs)
    
    def __repr__(self):
        """
        Print a string representation of the instance.
        """
        num_pops = len(self.pop_ids)
        num_bins = len(self.stats) - 1
        ret = f'H2stats with {num_pops} samples in {num_bins} bins'
        return ret

    def subset(self, to_pop_ids=None, min_r=None, max_r=None):
        """
        Subset the instance by population ids or bin indices.
        """
        bins, stats, covs = self.bins, self.stats, self.covs

        if to_pop_ids is not None:
            _pop_ids = self.pop_ids
            labels = get_stat_labels(_pop_ids)
            include = lambda pair: pair[0] in to_pop_ids and pair[1] in to_pop_ids
            idx = np.nonzero([include(pair) for pair in labels])[0]
            stats = stats[:, idx]
            covs = np.stack([cov[np.ix_(idx, idx)] for cov in covs])

            # sort pop_ids to preserve original order
            sorter = np.sort(np.searchsorted(_pop_ids, to_pop_ids))
            pop_ids = [_pop_ids[i] for i in sorter]

        if min_r is not None or max_r is not None:
            _bins = bins
            if min_r is None:
                min_r = 0
            if max_r is None:
                max_r = 0.5
            _bs = list(zip(_bins[:-1], _bins[1:]))
            min_bin = _bs.index(min_r)
            max_bin = _bs.index(max_r)
            bins = _bins[min_bin:max_bin + 1]
            if covs is not None:
                covs = np.vstack((covs[:-1][min_bin:max_bin], covs[-1][None]))
            stats = np.vstack((stats[:-1][min_bin:max_bin], stats[-1][None]))

        return H2stats(stats, bins=bins, pop_ids=pop_ids, covs=covs)
    
    def write(self, fname):
        #
        with open(fname, 'wb+') as file:
            pickle.dump(file, self)
        return 
    
    def write_dict(self, fname):
        #
        dic = self.asdict()
        with open(fname, 'wb+') as file:
            pickle.dump(file, dic)
        return 


_r_cache = {}


def get_r_points(bins):
    """
    Returns an array with edges from `bins` plus midpoints for interpolation.
    """
    if str(bins) in _r_cache:
        r_steps = _r_cache[bins]
    else:
        r_steps = np.sort(
            np.concatenate((bins, bins[:-1] + np.diff(bins) / 2))
        )
    return r_steps


def interpolate_quadratic(arr):
    """
    Applies Simpson's Rule to estimate expected H2 for each bin.
    """
    ret = 1/6 * arr[:-1:2] + 2/3 * arr[1::2] + 1/6 * arr[2::2]
    return ret


def get_stat_labels(pop_ids):
    """
    Get a list of tuples naming one and two-sample statistics from a list of
    population ids.
    """
    num_pops = len(pop_ids)
    labels = []
    for i in range(num_pops):
        for j in range(i, num_pops):
            labels.append((pop_ids[i], pop_ids[j]))
    return labels
