import demes

import demesdraw

from IPython.display import display

import matplotlib.pyplot as plt

import matplotlib

import msprime

import numpy as np

import time


if __name__ == "__main__":
    plt.rcParams['figure.dpi'] = 100
    matplotlib.use('Qt5Agg')


class DemogCluster:
    """
    Keeps a demography, tree sequences, and tree sequence statistics
    conveniently grouped together for comparison to other clusters.

    Parameters for ancestry simulations are defined in instantiation
    """

    def __init__(self, id, demog, n_trials, sample_pops, recomb_rate=1e-8,
                 mut_rate=1e-8, length=1e8, window_length=5e5, sample_size=1,
                 mutate=False, color=None):
        """
        Initialize a DemogCluster

        :param id: demography scenario name
        :param demog: msprime.Demography instance
        :param n_trials:
        :param sample_pops: specifies sample populations by string id
        :type sample_pops: list of str
        :param recomb_rate:
        :param mut_rate:
        :param length:
        :param window_length:
        :param sample_size:
        :param mutate:
        :param color: specify a color for plotting statistics from this cluster
        """
        sample_pops = list(set(sample_pops))
        self.id = id
        self.demog = demog
        pop_names = [pop.name for pop in demog.populations]
        pop_index = np.arange(demog.num_populations)
        self.name_to_pop = dict(zip(pop_names, pop_index))
        self.pop_to_name = dict(zip(pop_index, pop_names))
        self.n_trials = n_trials
        self.samples = dict(zip(sample_pops,
                                np.full(len(sample_pops), sample_size)))
        sample_id = np.sort([self.name_to_pop[name] for name in sample_pops])
        self.labels = [self.pop_to_name[id] for id in sample_id]
        self.sample_map = dict(zip(sample_id, sample_pops))
        self.n_sample_pops = len(sample_pops)
        self.recomb_rate = recomb_rate
        self.mutate = mutate
        self.mut_rate = mut_rate
        self.length = length
        self.window_length = window_length
        self.n_windows = int(length / window_length)
        self.color = None
        self.trials = []
        self.pi = None
        self.pi_xy = None
        self.f2 = None

    @classmethod
    def load_graph(cls, name, n_trials, sample_pops, path="c:/archaic/yamls/",
                   **kwargs):
        """
        Load a .yaml graph file and convert it to an msprime Demography
        instance to instantiate a DemogCluster

        :param name:
        :param n_trials:
        :param sample_pops:
        :param path:
        :param kwargs:
        :return:
        """
        filename = path + name
        graph = demes.load(filename)
        demog = msprime.Demography.from_demes(graph)
        return cls(name, demog, n_trials, sample_pops, *kwargs)

    def __repr__(self):
        Mb = int(self.length / 1e6)
        all = self.n_trials
        done = len(self.trials)
        return f"DemogCluster {self.id}, {Mb} Mb, {done} of {all} complete"

    @property
    def two_way_labels(self):
        labels = self.labels
        labels = [(labels[i], labels[j]) for i, j in self.two_way_index]
        return labels

    @property
    def two_way_index(self):
        n = self.n_sample_pops
        unique = int((np.square(n) - n) / 2)
        coords = []
        for i in np.arange(n):
            for j in np.arange(0, i):
                coords.append((i, j))
        return coords

    def plot_demog(self):
        graph = self.demog.to_demes()
        demesdraw.tubes(graph)

    def simulate(self, verbose=True):
        """
        Run self.n_trials coalescent simulations using msprime sim_ancestry
        function. If self.mutate == True, also simulate mutation
        """
        time0 = time.time()
        for i in np.arange(self.n_trials):
            multi_window = []
            for j in np.arange(self.n_windows):
                ts = msprime.sim_ancestry(samples=self.samples,
                                          demography=self.demog,
                                          ploidy=2,
                                          model="hudson",
                                          sequence_length=self.window_length,
                                          recombination_rate=self.recomb_rate
                                          )
                if self.mutate:
                    ts = msprime.sim_mutations(ts, rate=self.mut_rate)
                multi_window.append(ts)
            self.trials.append(multi_window)
        time1 = time.time()
        if verbose:
            t = np.round(time1 - time0, 2)
            print(f"{self.n_trials} trials w/ {self.length} bp set up in {t} s")

    def compute_diversity(self, verbose=True):
        """
        Compute the diversities for each trial

        :return:
        """
        time0 = time.time()
        pi = np.zeros((self.n_trials, len(self.sample_map)))
        for i, ts_iterator in enumerate(self.trials):
            pi[i] = self.compute_trial_diversity(ts_iterator)
        time1 = time.time()
        if verbose:
            t = np.round(time1 - time0, 2)
            Mb = int(self.length / 1e6)
            print(f"pi across {self.n_trials} {Mb} Mb trials computed in {t} s")
        self.pi = pi

    def compute_trial_diversity(self, trial):
        """
        Compute window diversities for one ts_iterator and return their mean

        :return:
        """
        trial_pi = np.zeros((self.n_windows, len(self.sample_map)))
        for i, window_ts in enumerate(trial):
            trial_pi[i] = self.compute_window_diversity(window_ts)
        mean_trial_pi = np.mean(trial_pi, axis=0)
        return mean_trial_pi

    def compute_window_diversity(self, window_ts):
        """
        Compute diversity for one window of a ts_iterator

        :param window_ts:
        :return:
        """
        window_pi = np.zeros(len(self.sample_map))
        for i, pop_id in enumerate(self.sample_map):
            sample = window_ts.samples(population=pop_id)
            window_pi[i] = window_ts.diversity(mode="branch",
                                               sample_sets=sample)
        window_pi *= self.mut_rate
        return window_pi

    def compute_divergence(self, verbose=True):
        """
        Compute the divergences across all 2-tuples

        :return:
        """
        time0 = time.time()
        pi_xy = np.zeros((self.n_trials, len(self.sample_map),
                          len(self.sample_map)))
        for i, trial in enumerate(self.trials):
            pi_xy[i] = self.compute_trial_divergence(trial)
        time1 = time.time()
        if verbose:
            t = np.round(time1 - time0, 2)
            Mb = int(self.length / 1e6)
            print(f"pi_xy across {self.n_trials} {Mb} Mb trials computed in {t} s")
        self.pi_xy = pi_xy

    def compute_trial_divergence(self, trial):
        """
        Compute divergence for one ts_iterator

        :return:
        """
        trial_pi_xy = np.zeros((self.n_windows, len(self.samples),
                                len(self.samples)))
        for i, window_ts in enumerate(trial):
            trial_pi_xy[i] = self.compute_window_divergence(window_ts)
        mean_trial_pi_xy = np.mean(trial_pi_xy, axis=0)
        return mean_trial_pi_xy

    def compute_window_divergence(self, window_ts):
        window_pi_xy = np.zeros((len(self.sample_map), len(self.sample_map)))
        for i, pop0_id in enumerate(self.sample_map):
            sample0 = window_ts.samples(population=pop0_id)
            for j, pop1_id in enumerate(self.sample_map):
                sample1 = window_ts.samples(population=pop1_id)
                window_pi_xy[i, j] = window_ts.divergence(
                    sample_sets=[sample0, sample1], mode="branch")
        window_pi_xy *= self.mut_rate
        return window_pi_xy

    def compute_f2(self, verbose=True):
        """
        Compute the f2 statistic across all 2-tuples

        :return:
        """
        time0 = time.time()
        f2 = np.zeros((self.n_trials, len(self.sample_map),
                       len(self.sample_map)))
        for i, trial in enumerate(self.trials):
            f2[i] = self.compute_trial_f2(trial)
        time1 = time.time()
        if verbose:
            t = np.round(time1 - time0, 2)
            Mb = int(self.length / 1e6)
            print(f"f2 across {self.n_trials} {Mb} Mb trials computed in {t} s")
        self.f2 = f2

    def compute_trial_f2(self, ts_iterator):
        n_pops = len(self.sample_map)
        trial_f2 = np.zeros((self.n_windows, n_pops, n_pops))
        for i, window_ts in enumerate(ts_iterator):
            trial_f2[i] = self.compute_window_f2(window_ts)
        mean_f2 = np.mean(trial_f2, axis=0)
        return mean_f2

    def compute_window_f2(self, window_ts):
        window_f2 = np.zeros((len(self.sample_map), len(self.sample_map)))
        for i, pop0_id in enumerate(self.sample_map):
            sample0 = window_ts.samples(population=pop0_id)
            for j, pop1_id in enumerate(self.sample_map):
                sample1 = window_ts.samples(population=pop1_id)
                window_f2[i, j] = window_ts.f2(sample_sets=[sample0, sample1],
                                               mode="branch")
        return window_f2

    def clear_ts(self):
        """
        remove tree structures from the instance

        :return:
        """
        self.trials = []


def load_graph(name, path="c:/archaic/yamls/"):
    filename = path + name
    graph = demes.load(filename)
    return graph


def graph_to_demog(graph):
    demog = msprime.Demography.from_demes(graph)
    return demog


def load_demog(name, path="c:/archaic/yamls/"):
    graph = load_graph(name, path=path)
    demog = graph_to_demog(graph)
    return demog


def plot_graph(graph, plot_ka=False):
    ax = demesdraw.tubes(graph)
    if plot_ka:
        ax_ = ax.twinx()
        max_gen = ax.get_ylim()[1]
        max_time = max_gen * 29 / 1000
        ax_.set_ylim(0, max_time)
        ax_.set_ylabel("time ago (ka)")
    return graph


def pi_scatter_plot(mean_pi):
    labels = ["Denisovan", "Neanderthal", "Modern African", "Modern European"]
    colors = ["green", "purple", "red", "blue"]
    npops = np.shape(mean_pi)[1]
    reps = len(mean_pi)
    fig = plt.figure(figsize=(8,6))
    sub = fig.add_subplot(111)
    for i in np.arange(npops):
        x = np.random.uniform(i - 0.1, i + 0.1, size=reps)
        plt.scatter(x, mean_pi[:, i], color=colors[i], label=labels[i],
                    marker='x')
    sub.set_xlim(-0.3, npops - 0.7)
    sub.set_ylim(0, )
    sub.set_ylabel("branch length diversity")
    sub.legend()
    fig.show()


def pi_violin_plot(mean_pi):
    npops = np.shape(mean_pi)[1]
    fig = plt.figure(figsize=(8,6))
    sub = fig.add_subplot(111)
    x = np.arange(npops)
    sub.violinplot(mean_pi, positions=x, widths=0.3, showmeans=True)
    sub.set_xlim(-1, npops)
    sub.set_ylim(0, )
    fig.show()


def div_violin_plot(div):
    """
    Divergence combinations. Order D, N, X, Y
    DD ND XD YD
    DN NN XN YN
    DX NX XX YX
    DY NY XY YY

    unique combinations: DN, DX, DY, NX, NY, XY

    :param mean_pi:
    :return:
    """
    ndivs = 6
    reps = len(div)
    fig = plt.figure(figsize=(8, 6))
    sub = fig.add_subplot(111)
    x = np.arange(ndivs)
    unique = np.zeros((reps, 6))
    unique[:, 0] = div[:, 1, 0]
    unique[:, 1] = div[:, 2, 0]
    unique[:, 2] = div[:, 3, 0]
    unique[:, 3] = div[:, 2, 1]
    unique[:, 4] = div[:, 3, 1]
    unique[:, 5] = div[:, 3, 2]
    sub.violinplot(unique, positions=x, widths=0.3, showmeans=True)
    sub.set_xlim(-1, ndivs)
    sub.set_xticks(np.arange(6), ["D-N", "D-X", "D-Y", "N-X", "N-Y", "X-Y"])
    sub.set_ylim(0, )
    fig.show()


def pi_box_plot(pi, info):
    """
    Plot an array of mean diversity values from n simulations using box plots
    with markers for mean values. n = len(mean_pi)

    :param pi:
    :return:
    """
    npops = info["npops"]
    n = info['n']
    fig = plt.figure(figsize=(8,6))
    sub = fig.add_subplot(111)
    x = np.arange(npops)
    colors = ["lightgreen", "gold", "red", "blue"]
    mean_style = dict(markerfacecolor="white", markeredgecolor='black',
                      marker="D")
    median_style = dict(linewidth=2, color='black')
    boxplot = sub.boxplot(pi, positions=x, widths=0.8, showmeans=True,
                          medianprops=median_style, patch_artist=True,
                          meanprops=mean_style)
    for box, color in zip(boxplot['boxes'], colors):
        box.set_facecolor(color)
    sub.set_xlim(-0.6, npops-0.4)
    sub.set_ylim(0, )
    demog = info["demog"]
    labels = [demog.populations[i].name for i in info["sample_pops"]]
    sub.set_xticks(np.arange(npops), labels)
    sub.set_ylabel("branch-length diversity")
    sub.set_xlabel("populations")
    tract = int(info["iter"] * info["window_size"] * 1e-6)
    sub.set_title(f"population diversities, n = {n}, {tract} Mb")
    fig.show()


def div_box_plot(div, info):
    """
    Divergence combinations. Order D, N, X, Y
    DD ND XD YD
    DN NN XN YN
    DX NX XX YX
    DY NY XY YY

    unique combinations: DN, DX, DY, NX, NY, XY

    :param mean_pi:
    :return:
    """
    ndivs = 6
    npops = info["npops"]
    n = info['n']
    fig = plt.figure(figsize=(8, 6))
    sub = fig.add_subplot(111)
    x = np.arange(ndivs)
    unique = np.zeros((n, 6))
    unique[:, 0] = div[:, 1, 0]
    unique[:, 1] = div[:, 2, 0]
    unique[:, 2] = div[:, 3, 0]
    unique[:, 3] = div[:, 2, 1]
    unique[:, 4] = div[:, 3, 1]
    unique[:, 5] = div[:, 3, 2]
    mean_style = dict(markerfacecolor="red", markeredgecolor='black',
                      marker="s")
    median_style = dict(linewidth=2, color='red')
    sub.boxplot(unique, positions=x, widths=0.8, showmeans=True,
                medianprops=median_style, meanprops=mean_style)
    sub.set_xlim(-1, ndivs)
    sub.set_xticks(np.arange(6), ["D-N", "D-X", "D-Y", "N-X", "N-Y", "X-Y"])
    sub.set_ylim(0, )
    sub.set_ylabel("branch-length diversity")
    sub.set_xlabel("populations")
    tract = int(info["iter"] * info["window_size"] * 1e-6)
    sub.set_title(f"population divergences n = {n}, {tract} Mb")
    fig.show()


def compare_one_way(statistic, *args):
    pass


def compare_two_way(statistic, *args):
    """
    Compare a two-way statisitc between demographic clusters given in *args.

    All clusters must have the same sample populations; otherwise they
    cannot be compared and the function will not work properly. Different
    numbers of trials are permitted.

    :param args:
    :return:
    """
    n_clusters = len(args)
    indices = args[0].two_way_index
    labels = args[0].two_way_labels
    n_two_ways = len(indices)
    stat_stack = [np.zeros((args[i].n_trials, n_two_ways))
                  for i in np.arange(n_clusters)]
    for k, cluster in enumerate(args):
        for r, i, j in enumerate(indices):
            stat_stack[k][:, r] = cluster.pi_xy[:, i, j]
    x_loc = np.arange(n_two_ways)
    mean_style = dict(markerfacecolor="red", markeredgecolor='black',
                      marker="s")
    median_style = dict(linewidth=2, color='red')
    fig = plt.figure(figsize=(8, 6))
    sub = fig.add_subplot(111)
    for i, cluster in enumerate(args):
        stats = stat_stack[i]
        sub.boxplot(stats, positions=x, widths=0.8, showmeans=True,
                    medianprops=median_style, meanprops=mean_style)
    sub.set_xlim(-1, n_two_ways)
    sub.set_xticks(np.arange(6), ["D-N", "D-X", "D-Y", "N-X", "N-Y", "X-Y"])
    sub.set_ylim(0, )
    sub.set_ylabel("branch-length diversity")
    sub.set_xlabel("populations")
    sub.set_title(f"population divergences n = {10}, {10} Mb")
    fig.show()











x = DemogCluster.load_graph("achenbach.yaml", 2, ['N', 'D', 'X', "Y"])
