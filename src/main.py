import demes

import demesdraw

from IPython.display import display

import matplotlib.pyplot as plt

import matplotlib

import msprime

import numpy as np

import time

import os


if __name__ == "__main__":
    plt.rcParams['figure.dpi'] = 100
    matplotlib.use('Qt5Agg')


class Cluster:
    """
    Keeps a demography, tree sequences, and tree sequence statistics
    conveniently grouped together for comparison to other clusters.

    Parameters for ancestry simulations are defined in instantiation
    """

    def __init__(self, name, demography, n_reps, sample_pops, recomb_rate=1e-8,
                 mut_rate=1e-8, seq_length=1e8, window_length=5e5, sample_size=1,
                 source=None, statistics=None, created=None):
        """
        Initialize a DemogCluster

        :param id: demography scenario name
        :param demog: msprime.Demography instance
        :param n_reps:
        :param sample_pops: specifies sample populations by string id
        :type sample_pops: list of str
        :param recomb_rate:
        :param mut_rate:
        :param length:
        :param window_length:
        :param sample_size: number of diploid individuals to sample per pop
        """
        self.name = name
        if source:
            self.source = source
        self.demography = demography
        self.n_reps = n_reps
        self.sample_size = sample_size
        #
        sample_pops = list(set(sample_pops))
        self.dim = len(sample_pops)
        full_pop_names = [pop.name for pop in demography.populations]
        pop_ids = np.arange(demography.num_populations)
        self.name_id_map = dict(zip(full_pop_names, pop_ids))
        self.id_name_map = dict(zip(pop_ids, full_pop_names))
        sample_ids = [self.name_id_map[name] for name in sample_pops]
        sample_ids.sort()
        self.sample_ids = np.array(sample_ids)
        sample_sizes = [sample_size] * self.dim
        self.sample_dict = dict(zip(sample_pops, sample_sizes))
        order = [self.name_id_map[name] for name in sample_pops]
        sorter = np.argsort(order)
        sorted_sample_pops = [sample_pops[i] for i in sorter]
        self.labels = dict(zip(np.arange(self.dim), sorted_sample_pops))
        self.recomb_rate = recomb_rate
        self.mut_rate = mut_rate
        self.seq_length = seq_length
        self.window_length = window_length
        self.n_windows = int(seq_length / window_length)
        self.replicates = []
        self.statistics = {}
        if statistics:
            self.statistics = statistics

    @classmethod
    def load_graph(cls, source, n_trials, sample_pops, path="c:/archaic/yamls/",
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
        filename = path + source
        graph = demes.load(filename)
        demog = msprime.Demography.from_demes(graph)
        name = source.replace(".yaml", "")
        return cls(name, demog, n_trials, sample_pops, source=source, **kwargs)

    @classmethod
    def load_data(cls, dir_name, path="c:/archaic/statistics/"):
        full_path = path + dir_name
        filenames = os.listdir(path + dir_name)
        arrays = []
        names = []
        for name in filenames:
            arrays.append(np.loadtxt(full_path + '/' + name))
            names.append(name.replace(".txt", ''))
        statistics = dict(zip(names, arrays))
        for key in statistics:
            if key in ["pi_xy", "f2", "Fst"]:
                arr = statistics[key]
                i, jk = arr.shape
                j = int(np.sqrt(jk))
                statistics[key] = arr.reshape(i, j, j)
        dicts = []
        for name in filenames:
            file = open(full_path + '/' + name, 'r')
            lines = [line.replace('#', '').replace("\n", '')
                     for line in file if '#' in line]
            file.close()
            dic = eval("{" + ",".join(lines) + "}")
            dicts.append(dic)
        for dic in dicts:
            if dic != dicts[0]:
                raise ValueError("incompatible headers in data set!")
        d = dicts[0]
        filename = "c:/archaic/yamls/" + d["source"]
        graph = demes.load(filename)
        demog = msprime.Demography.from_demes(graph)
        return cls(name=d["name"], demography=demog, n_reps=d["n_reps"],
                   sample_pops=d["sample_pops"], recomb_rate=d["recomb_rate"],
                   mut_rate=d["mut_rate"], seq_length=d["seq_length"],
                   window_length=d["window_length"],
                   sample_size=d["sample_size"],  source=d["source"],
                   statistics=statistics)

    @property
    def pi(self):
        if "pi" not in self.statistics.keys():
            raise KeyError("pi has not yet been computed!")
        return self.statistics["pi"]

    @property
    def pi_xy(self):
        if "pi_xy" not in self.statistics.keys():
            raise KeyError("pi_xy has not yet been computed!")
        return self.statistics["pi_xy"]

    @property
    def f2(self):
        if "f2" not in self.statistics.keys():
            raise KeyError("f2 has not yet been computed!")
        return self.statistics["f2"]

    @property
    def Fst(self):
        if "Fst" not in self.statistics.keys():
            raise KeyError("Fst has not yet been computed!")
        return self.statistics["Fst"]

    @property
    def two_way_labels(self):
        labels = self.labels
        labels = [(labels[i], labels[j]) for i, j in self.two_way_index]
        return labels

    @property
    def two_way_index(self):
        unique = int((np.square(self.dim) - self.dim) / 2)
        coords = []
        for i in np.arange(self.dim):
            for j in np.arange(0, i):
                coords.append((i, j))
        return coords

    @property
    def seq_Mb(self):
        return int(self.seq_length * 1e-6)

    def plot_demography(self, log_time=False):
        graph = self.demography.to_demes()
        demesdraw.tubes(graph, log_time=log_time)

    def simulate(self, verbose=True):
        """
        Run self.n_trials coalescent simulations using msprime sim_ancestry
        function.
        """
        time0 = time.time()
        for i in np.arange(self.n_reps):
            rep = Replicate(self.n_windows, self.window_length,
                            self.recomb_rate, self.demography,
                            self.sample_dict, self.mut_rate, self.sample_ids)
            self.replicates.append(rep)
        time1 = time.time()
        if verbose:
            t = np.round(time1 - time0, 2)
            print(f"{self.n_reps} reps, {self.seq_length} bp simulated in {t} s")

    def compute_pi(self):
        """
        Compute the population diversities in each replicate

        :return:
        """
        pi = np.zeros((self.n_reps, self.dim))
        for i, replicate in enumerate(self.replicates):
            pi[i] = replicate.compute_pi()
        self.statistics['pi'] = pi

    def compute_pi_xy(self):
        """
        Compute the divergences across all 2-tuples

        :return:
        """
        pi_xy = np.zeros((self.n_reps, self.dim, self.dim))
        for i, replicate in enumerate(self.replicates):
            pi_xy[i] = replicate.compute_pi_xy()
        self.statistics['pi_xy'] = pi_xy

    def compute_f2(self):
        """
        Compute the f2 statistic across all 2-tuples

        :return:
        """
        f2 = np.zeros((self.n_reps, self.dim, self.dim))
        for i, replicate in enumerate(self.replicates):
            f2[i] = replicate.compute_f2()
        self.statistics['f2'] = f2

    def compute_Fst(self):
        """
        Compute the f2 statistic across all 2-tuples

        :return:
        """
        Fst = np.zeros((self.n_reps, self.dim, self.dim))
        for i, replicate in enumerate(self.replicates):
            Fst[i] = replicate.compute_Fst()
        self.statistics['Fst'] = Fst

    def clear_replicates(self):
        """
        Delete all replicates

        :return:
        """
        self.replicates = []

    def get_header(self):
        items = [f"'name' : '{self.name}'",
                 f"'source' : '{self.source}'",
                 f"'sample_pops' : {list(self.sample_dict.keys())}",
                 f"'sample_size' : {self.sample_size}",
                 f"'n_reps' : {self.n_reps}",
                 f"'n_windows' : {self.n_windows}",
                 f"'seq_length' : {self.seq_length}",
                 f"'window_length' : {self.window_length}",
                 f"'recomb_rate' : {self.recomb_rate}",
                 f"'mut_rate' : {self.mut_rate}",
                 f"'created' : {None}"
                 ]
        header = "\n".join(items)
        return header

    def write(self, dir_name=None, path="c:/archaic/statistics/"):
        """
        Save statistics in .npz format

        :param filename:
        :return:
        """
        if not dir_name:
            dir_name = self.name +'/'
        full_path = os.path.join(path, dir_name)
        check = os.path.exists(full_path)
        while check:
            dir_name = dir_name.replace("/", "0/")
            full_path = os.path.join(path, dir_name)
            check = os.path.exists(full_path)
        os.mkdir(full_path)
        header = self.get_header()
        for statistic in self.statistics:
            filename = os.path.join(full_path, statistic + ".txt")
            arr = self.statistics[statistic]
            if arr.ndim == 3:
                i, j, k = arr.shape
                arr = arr.reshape(i, j * k)
            np.savetxt(filename, arr, header=header)
        return full_path



class Replicate:
    """
    Container for multiple Windows class instances. Immediately instantiates
    and simulates its windows.
    """
    def __init__(self, n_windows, window_length, recomb_rate, demography,
                 sample_dict, mut_rate, sample_ids):
        """


        :param n_windows:
        :param window_length:
        :param recomb_rate:
        :param demography:
        :param sample_dict:
        :param mut_rate:
        """
        self.n_windows = n_windows
        self.window_length = window_length
        self.recomb_rate = recomb_rate
        self.demography = demography
        self.windows = []
        for i in np.arange(n_windows):
            self.windows.append(Window(window_length, recomb_rate, demography,
                                       sample_dict, mut_rate, sample_ids))
        self.dim = len(sample_dict)

    def compute_pi(self):
        """
        Compute the population diversities in each window and return the mean

        :return:
        """
        pi_arr = np.zeros((self.n_windows, self.dim))
        for i, window in enumerate(self.windows):
            pi_arr[i] = window.pop_pi
        pi = np.mean(pi_arr, axis=0)
        return pi

    def compute_pi_xy(self):
        """
        Return the mean divergence across windows

        :return:
        """
        pi_xy = np.zeros((self.n_windows, self.dim, self.dim))
        for i, window in enumerate(self.windows):
            pi_xy[i] = window.pi_xy
        f2 = np.mean(pi_xy, axis=0)
        return f2

    def compute_f2(self):
        """
        Return the mean f2 statistic across windows

        :return:
        """
        f2_arr = np.zeros((self.n_windows, self.dim, self.dim))
        for i, window in enumerate(self.windows):
            f2_arr[i] = window.f2
        f2 = np.mean(f2_arr, axis=0)
        return f2

    def compute_Fst(self):
        """
        Return the mean F_st statistic across windows

        :return:
        """
        Fst_arr = np.zeros((self.n_windows, self.dim, self.dim))
        for i, window in enumerate(self.windows):
            Fst_arr[i] = window.Fst
        Fst = np.mean(Fst_arr, axis=0)
        return Fst


class Window:
    """
    Wrapper class for a single tree sequence. Instantiation immediately
    begins the coalescent simulation.
    """
    def __init__(self, window_length, recomb_rate, demography, sample_dict,
                 mut_rate, sample_ids):
        """


        :param window_length:
        :param recomb_rate:
        :param demography:
        :param sample_dict:
        :param mut_rate:
        """
        self.window_length = window_length
        self.recomb_rate = recomb_rate
        self.demography = demography
        self.sample_dict = sample_dict
        self.mut_rate = mut_rate
        self.sample_ids = sample_ids
        self.tree_seq = msprime.sim_ancestry(samples=sample_dict,
                                             demography=demography,
                                             ploidy=2,
                                             model="hudson",
                                             sequence_length=window_length,
                                             recombination_rate=recomb_rate
                                             )
        self.dim = len(sample_dict)  # dimension

    @property
    def pi(self):
        """
        Compute diversity

        :return:
        """
        samples = self.tree_seq.samples()
        pi = self.tree_seq.diversity(sample_sets=samples)
        pi *= self.mut_rate
        return pi

    @property
    def pop_pi(self):
        """
        Compute diversity in each sample population

        :return:
        """
        pi = np.zeros(self.dim)
        for i, pop_id in enumerate(self.sample_ids):
            sample = self.tree_seq.samples(population=pop_id)
            pi[i] = self.tree_seq.diversity(mode="branch", sample_sets=sample)
        pi *= self.mut_rate
        return pi

    @property
    def pi_xy(self):
        """
        Compute window divergence

        :return:
        """
        pi_xy = np.zeros((self.dim, self.dim))
        for i, pop0_id in enumerate(self.sample_ids):
            sample0 = self.tree_seq.samples(population=pop0_id)
            for j, pop1_id in enumerate(self.sample_ids):
                if j < i:
                    sample1 = self.tree_seq.samples(population=pop1_id)
                    pi_xy[i, j] = self.tree_seq.divergence(
                        sample_sets=[sample0, sample1], mode="branch")
        pi_xy *= self.mut_rate
        return pi_xy

    @property
    def f2(self):
        """
        Compute window f2

        :return:
        """
        f2 = np.zeros((self.dim, self.dim))
        for i, pop0_id in enumerate(self.sample_ids):
            sample0 = self.tree_seq.samples(population=pop0_id)
            for j, pop1_id in enumerate(self.sample_ids):
                if j < i:
                    sample1 = self.tree_seq.samples(population=pop1_id)
                    f2[i, j] = self.tree_seq.f2(sample_sets=[sample0, sample1],
                                                mode="branch")
        f2 *= self.mut_rate
        return f2

    @property
    def Fst(self):
        """
        Compute window F_st

        :return:
        """
        Fst = np.zeros((self.dim, self.dim))
        for i, pop0_id in enumerate(self.sample_ids):
            sample0 = self.tree_seq.samples(population=pop0_id)
            for j, pop1_id in enumerate(self.sample_ids):
                if j < i:
                    sample1 = self.tree_seq.samples(population=pop1_id)
                    Fst[i, j] = self.tree_seq.Fst(
                        sample_sets=[sample0, sample1], mode="branch")
        return Fst


def one_way_1d_plot(statistic, *args):
    """
    Plot a box plot comparison of a one way statistic from several clusters

    :param statistic:
    :param args:
    :return:
    """
    n_clusters = len(args)
    labels = args[0].labels
    n_pops = args[0].n_sample_pops
    stat_stack = [np.zeros((args[i].n_trials, n_pops))
                  for i in np.arange(n_clusters)]
    for i, cluster in enumerate(args):
        stat_stack[i][:] = cluster.pi
    x_loc = np.arange(n_pops)
    colors = []
    fig = plt.figure(figsize=(8, 6))
    sub = fig.add_subplot(111)
    width = 0.5 / n_clusters
    pos = np.linspace(-0.5 + width * 1.25, 0.5 - width * 1.25, n_clusters)
    for i, cluster in enumerate(args):
        if cluster.color:
            color = cluster.color
        else:
            color = "white"
        colors.append(color)
        median_style = dict(linewidth=1, color="black")
        stats = stat_stack[i]
        x = x_loc + pos[i]
        b = sub.boxplot(stats, positions=x, widths=width, capwidths=0,
                        medianprops=median_style, patch_artist=True)
        for box in b['boxes']:
            box.set(facecolor=color)
    sub.set_xlim(-1, n_pops)
    sub.set_xticks(np.arange(n_pops), labels)
    sub.set_ylim(0, )
    sub.set_ylabel(f"{statistic}")
    sub.set_xlabel("population")
    n = int(np.mean([cluster.n_trials for cluster in args]))
    Mb = args[0].length * 1e-6
    sub.set_title(f"{statistic} in {n_clusters} trials, n = {n}, {Mb} Mb")
    sub.legend(handles=[matplotlib.patches.Patch(color=colors[i], label=args[i].id)
                for i in np.arange(n_clusters)])
    fig.show()


def two_way_1d(statistic, *args):
    """
    Compare a two-way statistic between demographic clusters given in *args.

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
        for r, (i, j) in enumerate(indices):
            if statistic == "pi_xy" or statistic == "divergence":
                stat_stack[k][:, r] = cluster.pi_xy[:, i, j]
            elif statistic == "f2":
                stat_stack[k][:, r] = cluster.f2[:, i, j]
    x_loc = np.arange(n_two_ways)
    colors = []
    fig = plt.figure(figsize=(8, 6))
    sub = fig.add_subplot(111)
    width = 0.5 / n_clusters
    pos = np.linspace(-0.5 + width * 1.25, 0.5 - width * 1.25, n_clusters)
    for i, cluster in enumerate(args):
        if cluster.color:
            color = cluster.color
        else:
            color = "white"
        colors.append(color)
        median_style = dict(linewidth=1, color="black")
        stats = stat_stack[i]
        x = x_loc + pos[i]
        b = sub.boxplot(stats, positions=x, widths=width, capwidths=0,
                        medianprops=median_style, patch_artist=True)
        for box in b['boxes']:
            box.set(facecolor=color)
    sub.set_xlim(-1, n_two_ways)
    dash_labels = [f"{pop1}-{pop2}" for pop1, pop2 in labels]
    sub.set_xticks(np.arange(n_two_ways), dash_labels)
    sub.set_ylim(0, )
    sub.set_ylabel(f"{statistic}")
    sub.set_xlabel("populations")
    n = int(np.mean([cluster.n_trials for cluster in args]))
    Mb = args[0].length * 1e-6
    sub.set_title(f"{statistic} in {n_clusters} trials, n = {n}, {Mb} Mb")
    sub.legend(handles=[matplotlib.patches.Patch(color=colors[i], label=args[i].id)
                for i in np.arange(n_clusters)])
    fig.show()
    # add color, legend, labels etc


def violins(*args, statistic="pi"):
    """
    plots overlapping violin plots

    :param args: lists of clusters. each list is displayed in one subplot
    :return:
    """
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111)
    colors = ["green", "gold", "red", "blue"]
    median_style = dict(linewidth=1, color="black")
    dim = args[0][0].dim
    for i, clusters in enumerate(args):
        n_clusters = len(clusters)
        alphas = np.linspace(1, 0.4, n_clusters)
        for j, cluster in enumerate(clusters):
            stat = cluster.statistics[statistic]
            x = np.arange(0, dim) + i * (dim + 0.5)
            v = ax.violinplot(stat, positions=x, widths=1, showmeans=False,
                               showmedians=False, showextrema=False)
            b = ax.boxplot(stat, positions=x, widths=0.25, capwidths=0,
                            medianprops=median_style, patch_artist=True)
            for box, color in zip(b['boxes'], colors):
                box.set(facecolor=color)
            for part, color in zip(v['bodies'], colors):
                part.set_facecolor(color)
                part.set_edgecolor('black')
                part.set_alpha(0.6)
    ax.set_ylim(0, )


def ranked(*cluster_groups, statistic="pi"):
    """
    plots side by side box plots

    :param cluster_groups:
    :param statistic:
    :return:
    """
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111)
    colors = ["green", "gold", "red", "blue"]
    median_style = dict(linewidth=1, color="black")
    dim = cluster_groups[0][0].dim
    x_offset = 0
    for i, clusters in enumerate(cluster_groups):
        n_clusters = len(clusters)
        alphas = np.linspace(1, 0.4, n_clusters)
        x = np.arange(0, dim * n_clusters, n_clusters)
        x += x_offset
        for j, cluster in enumerate(clusters):
            stat = cluster.statistics[statistic]
            b = ax.boxplot(stat, positions=x, widths=1, capwidths=0,
                            medianprops=median_style, patch_artist=True)
            for box, color in zip(b['boxes'], colors):
                box.set(facecolor=color, alpha=alphas[j])
            x += 1
        x_offset += dim * n_clusters
    ax.set_ylim(0, )
    ax.set_yticks(np.linspace(0, 0.001, 11))
    labels = list(cluster_groups[0][0].labels.values())
    ax.legend(handles=[matplotlib.patches.Patch(
        color=colors[i], label=labels[i]) for i in np.arange(dim)])


def project(cluster, x_axis, y_axis):
    """
    return mass migration proportions from a cluster

    recall that mass migrations are REVERSED in msprime!!!!

    :param cluster:
    :param x_axis: (source, dest)
    :param y_axis:
    :return:
    """
    x = 0
    y = 0
    events = cluster.demography.events
    for event in events:
        if "source" in event.asdict():
            if event.source == x_axis[1] and event.dest == x_axis[0]:
                x = event.proportion
            if event.source == y_axis[1] and event.dest == y_axis[0]:
                y = event.proportion
    return x, y


def compare_two_wayv(statistic, *args):
    """
    Compare a two-way statistic between demographic clusters given in *args.

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
        for r, (i, j) in enumerate(indices):
            if statistic == "pi_xy" or statistic == "divergence":
                stat_stack[k][:, r] = cluster.pi_xy[:, i, j]
            elif statistic == "f2":
                stat_stack[k][:, r] = cluster.f2[:, i, j]
    x_loc = np.arange(n_two_ways)
    colors = []
    fig = plt.figure(figsize=(8, 6))
    sub = fig.add_subplot(111)
    width = 0.5 / n_clusters
    pos = np.linspace(-0.5 + width * 1.15, 0.5 - width * 1.15, n_clusters)
    for i, cluster in enumerate(args):
        if cluster.color:
            color = cluster.color
        else:
            color = "black"
        colors.append(color)
        stats = stat_stack[i]
        x = x_loc + pos[i]
        v = sub.violinplot(stats, positions=x, widths=width, showmeans=True,
                           showmedians=True)
        for part in v['bodies']:
            part.set_facecolor(color)
            part.set_edgecolor('black')
            part.set_alpha(0.8)
    sub.set_xlim(-1, n_two_ways)
    dash_labels = [f"{pop1}-{pop2}" for pop1, pop2 in labels]
    sub.set_xticks(np.arange(n_two_ways), dash_labels)
    sub.set_ylim(0, )
    sub.set_ylabel(f"{statistic}")
    sub.set_xlabel("populations")
    n = int(np.mean([cluster.n_trials for cluster in args]))
    Mb = args[0].length * 1e-6
    sub.set_title(f"{statistic} in {n_clusters} trials, n = {n}, {Mb} Mb")
    sub.legend(handles=[matplotlib.patches.Patch(color=colors[i], label=args[i].id)
                for i in np.arange(n_clusters)])
    fig.show()


def one_way_violin(statistic, *args):
    n_clusters = len(args)
    labels = args[0].labels
    n_pops = args[0].n_sample_pops
    stat_stack = [np.zeros((args[i].n_trials, n_pops))
                  for i in np.arange(n_clusters)]
    for i, cluster in enumerate(args):
        if statistic == "pi":
            stat_stack[i][:] = cluster.pi
    x_loc = np.arange(n_pops)
    colors = []
    fig = plt.figure(figsize=(8, 6))
    sub = fig.add_subplot(111)
    width = 0.5 / n_clusters
    pos = np.linspace(-0.5 + width * 1.25, 0.5 - width * 1.25, n_clusters)
    for i, cluster in enumerate(args):
        if cluster.color:
            color = cluster.color
        else:
            color = "black"
        colors.append(color)
        stats = stat_stack[i]
        x = x_loc + pos[i]
        v = sub.violinplot(stats, positions=x, widths=width)
        means = np.mean(stats, axis=0)
        for partname in ('cbars', 'cmins', 'cmaxes'):
            part = v[partname]
            part.set_edgecolor("black")
            part.set_linewidth(1)
        for part in v['bodies']:
            part.set_facecolor(color)
            part.set_edgecolor('black')
            part.set_alpha(0.7)
        sub.scatter(x, means, color="black", marker='x')
    sub.set_xlim(-1, n_pops)
    sub.set_xticks(np.arange(n_pops), labels)
    sub.set_ylim(0, )
    sub.set_ylabel(f"{statistic}")
    sub.set_xlabel("population")
    n = int(np.mean([cluster.n_trials for cluster in args]))
    Mb = args[0].length * 1e-6
    sub.set_title(f"{statistic} in {n_clusters} trials, n = {n}, {Mb} Mb")
    sub.legend(handles=[matplotlib.patches.Patch(color=colors[i], label=args[i].id)
                for i in np.arange(n_clusters)], loc="best")
    fig.show()


def one_way_2d(var0, var1, label0, label1, clusters, statistic="pi",
               y_max=1e-3, y_tick=1e-4, title=None):
    """
    Plot projections of points in the space defined by two variables var0 and
    var1 and a statistic.

    :param var0:
    :param var1:
    :param label0:
    :param label1:
    :param clusters:
    :param statistic:
    :return:
    """
    colors = ["green", "orange", "red", "blue"]
    ids = clusters[0].sample_ids
    id_name_map = clusters[0].id_name_map
    names = [id_name_map[i] for i in ids]
    means = []
    stds = []
    x0s = []
    x1s = []
    dim = clusters[0].dim
    for cluster in clusters:
        _x0, _x1 = project(cluster, var0, var1)
        x0s.append(_x0)
        x1s.append(_x1)
        stats = cluster.statistics[statistic]
        means.append(np.mean(stats, axis=0))
        stds.append(np.std(stats, axis=0))
    x0s = np.array(x0s) # vector of x0 point for each cluster
    x1s = np.array(x1s) # vector of x1 point for each cluster
    means = np.array(means) # n_cluster * dim array of means
    stds = np.array(stds) # n_cluster * dim array of stds
    unique_x0s = np.sort(np.array(list(set(list(x0s)))))
    unique_x1s = np.sort(np.array(list(set(list(x1s)))))

    fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(11, 6), sharey='all')
    ax0, ax1 = axs

    for i, x0 in enumerate(unique_x0s):

        for j, x1 in enumerate(unique_x1s):

            line_widths = np.arange(1, 2 + len(unique_x1s))

            for k in np.arange(dim):
                y = means[:, k]
                err = stds[:, k]
                mask0 = np.nonzero(x0s == x0)
                mask1 = np.nonzero(x1s == x1)
                ax0.plot(x0s[mask1], y[mask1], color=colors[k],
                         linewidth=line_widths[j])  # horizontal
                ax0.plot(x0s[mask0], y[mask0], color=colors[k])  # vertical
                ax1.plot(x1s[mask0], y[mask0], color=colors[k])  # horizontal
                ax1.plot(x1s[mask1], y[mask1], color=colors[k])  # vertical
                if i == 0 and j == 0:
                    ax0.errorbar(x0s[mask1], y[mask1], color=colors[k], capsize=3,
                                 marker='+', yerr=err[mask1], label=names[k])
                    ax1.errorbar(x1s[mask0], y[mask0], color=colors[k], capsize=3,
                                 marker='+', yerr=err[mask0])
                else:
                    ax0.errorbar(x0s[mask1], y[mask1], color=colors[k], capsize=3,
                                 marker='+', yerr=err[mask1])
                    ax1.errorbar(x1s[mask0], y[mask0], color=colors[k], capsize=3,
                                 marker='+', yerr=err[mask0])
    ax0.set_xlabel(f"{label0} prop: {var0[0]} -> {var0[1]}")
    ax1.set_xlabel(f"{label1} prop: {var1[0]} -> {var1[1]}")
    ax0.set_title(f"projection to {label0}")
    ax1.set_title(f"projection to {label1}")
    ax0.set_ylabel(statistic)
    ax0.set_yticks(np.arange(0, y_max + y_tick, y_tick))
    n = clusters[0].n_reps
    Mb = clusters[0].seq_Mb
    if title:
        fig.suptitle(title + f": n = {n} reps / demography, {Mb} Mb / rep")
    fig.subplots_adjust(wspace=0.03, hspace=0)
    fig.legend()
    fig.show()


def two_way_2d(var0, var1, label0, label1, clusters, statistic="f2",
               y_max=1e-3, y_tick=1e-4, title=None):
    """
    Plot projections of points in the space defined by two variables var0 and
    var1 and a two-way statistic.

    :param var0:
    :param var1:
    :param label0:
    :param label1:
    :param clusters:
    :param statistic:
    :return:
    """
    colors = ["red", "blue", "gold", "green", "magenta", "purple"]
    labels = clusters[0].two_way_labels
    means = []
    stds = []
    x0s = []
    x1s = []
    dim = len(clusters[0].two_way_index)
    for cluster in clusters:
        _x0, _x1 = project(cluster, var0, var1)
        x0s.append(_x0)
        x1s.append(_x1)
        idx = cluster.two_way_index
        arr = cluster.statistics[statistic]
        stats = [arr[:, j, k] for j, k in idx]
        means.append(np.mean(stats, axis=1))
        stds.append(np.std(stats, axis=1))
    x0s = np.array(x0s) # vector of x0 point for each cluster
    x1s = np.array(x1s) # vector of x1 point for each cluster
    means = np.array(means) # n_cluster * dim array of means
    stds = np.array(stds) # n_cluster * dim array of stds
    unique_x0s = np.sort(np.array(list(set(list(x0s)))))
    unique_x1s = np.sort(np.array(list(set(list(x1s)))))

    fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(12, 6), sharey='all')
    ax0, ax1 = axs

    for i, x0 in enumerate(unique_x0s):

        for j, x1 in enumerate(unique_x1s):

            line_widths = np.arange(1, 2 + len(unique_x1s))

            for k in np.arange(dim):
                y = means[:, k]
                err = stds[:, k]
                mask0 = np.nonzero(x0s == x0)
                mask1 = np.nonzero(x1s == x1)
                ax0.plot(x0s[mask1], y[mask1], color=colors[k],
                         linewidth=line_widths[j])  # horizontal
                ax0.plot(x0s[mask0], y[mask0], color=colors[k])  # vertical
                ax1.plot(x1s[mask0], y[mask0], color=colors[k])  # horizontal
                ax1.plot(x1s[mask1], y[mask1], color=colors[k])  # vertical
                if i == 0 and j == 0:
                    ax0.errorbar(x0s[mask1], y[mask1], color=colors[k], capsize=3,
                                 marker='+', yerr=err[mask1], label=labels[k])
                    ax1.errorbar(x1s[mask0], y[mask0], color=colors[k], capsize=3,
                                 marker='+', yerr=err[mask0])
                else:
                    ax0.errorbar(x0s[mask1], y[mask1], color=colors[k], capsize=3,
                                 marker='+', yerr=err[mask1])
                    ax1.errorbar(x1s[mask0], y[mask0], color=colors[k], capsize=3,
                                 marker='+', yerr=err[mask0])
    ax0.set_xlabel(f"{label0}: {var0[0]} -> {var0[1]}")
    ax1.set_xlabel(f"{label1}: {var1[0]} -> {var1[1]}")
    ax0.set_title(f"projection to {label0}")
    ax1.set_title(f"projection to {label1}")
    ax0.set_ylabel(statistic)
    ax0.set_yticks(np.arange(0, y_max + y_tick, y_tick))
    n = clusters[0].n_reps
    Mb = clusters[0].seq_Mb
    if title:
        fig.suptitle(title + f": n = {n} reps / demography, {Mb} Mb / rep")
    fig.subplots_adjust(wspace=0.03, hspace=0)
    fig.legend()
    fig.show()


def simulate_and_write(filename, n_reps=100):
    print(f"est time: {n_reps * 3.5} s")
    null = Cluster.load_graph(filename, n_reps, ['N', 'D', 'X', "Y"])
    null.simulate()
    null.compute_pi()
    null.compute_pi_xy()
    null.compute_f2()
    null.compute_Fst()
    null.write()


a02_b02_names = ['rogers_a02_b02_d00_g00',
                 'rogers_a02_b02_d00_g03',
                 'rogers_a02_b02_d00_g06',
                 'rogers_a02_b02_d03_g00',
                 'rogers_a02_b02_d03_g03',
                 'rogers_a02_b02_d03_g06',
                 'rogers_a02_b02_d05_g00',
                 'rogers_a02_b02_d05_g03',
                 'rogers_a02_b02_d05_g06']


a02_b00_names = ['rogers_a02_b00_d00_g00',
                 'rogers_a02_b00_d00_g03',
                 'rogers_a02_b00_d00_g06',
                 'rogers_a02_b00_d03_g00',
                 'rogers_a02_b00_d03_g03',
                 'rogers_a02_b00_d03_g06',
                 'rogers_a02_b00_d05_g00',
                 'rogers_a02_b00_d05_g03',
                 'rogers_a02_b00_d05_g06']


a02_b02_clusters = [Cluster.load_data(name) for name in a02_b02_names]
a02_b00_clusters = [Cluster.load_data(name) for name in a02_b00_names]

# set up 2 more cluster groups with different beta proportions









