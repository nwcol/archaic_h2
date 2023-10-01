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


def load_yaml(filename = "c:/archaic/yamls/super_archaic.yaml"):
    graph = demes.load(filename)
    return graph


def plot_graph(graph, plot_ka=False):
    ax = demesdraw.tubes(graph)
    if plot_ka:
        ax_ = ax.twinx()
        max_gen = ax.get_ylim()[1]
        max_time = max_gen * 29 / 1000
        ax_.set_ylim(0, max_time)
        ax_.set_ylabel("time ago (ka)")
    return graph


def graph_to_demog(graph):
    demog = msprime.Demography.from_demes(graph)
    return demog


def sim_ancestry(demog, recomb_rate=1e-8, seq_length=1e4, reps=None, n=1):
    """
    Run a coalescent simulation using msprime sim_ancestry function

    :param demog:
    :param recomb_rate:
    :param seq_length:
    :param reps: n replicates to simulate
    :param n: sample size per sample population
    :return:
    """
    time0 = time.time()
    ts = msprime.sim_ancestry(samples={"X": n, "Y": n, "N": n, "D": n},
                              demography=demog, ploidy=2, model="hudson",
                              sequence_length=seq_length,
                              recombination_rate=recomb_rate,
                              num_replicates=reps)
    time1 = time.time()
    t = np.round(time1 - time0, 2)
    print(f"{n} {seq_length} bp windows in {t} s")
    return ts


def sim_mutations(ts, u=1e-8):
    time0 = time.time()
    mts = msprime.sim_mutations(ts, rate=u)
    time1 = time.time()
    t = np.round(time1 - time0, 2)
    print(f"in {t} s")
    return mts


def print_text(ts, demog):
    node_pop = ts.nodes_population
    names = [population.name for population in demog.populations]
    ids = np.arange(len(node_pop))
    node_names = {i : f"{i}:{names[node_pop[i]]}" for i in ids}
    print(ts.draw_text(node_labels=node_names))


def write_svg(ts, demog, name, path="svgs/", lim=[0,2000]):
    node_pop = ts.nodes_population
    names = [population.name for population in demog.populations]
    ids = np.arange(len(node_pop))
    node_names = {i : f"{i}:{names[node_pop[i]]}" for i in ids}
    svg_string = ts.draw_svg(path=path + name + ".svg",
                             size=[1400, 500],
                             y_axis=True,
                             y_gridlines=True,
                             y_ticks=np.arange(0, 3.2e5, 2e4),
                             x_scale="treewise",
                             x_lim=lim,
                             node_labels=node_names)


def get_sample_populations(ts):
    """
    Return the populations which are included in a tree sequences sample

    :param ts:
    :return:
    """
    sample_nodes = ts.samples()
    sample_pops = ts.nodes_population[sample_nodes]
    sample_pops = np.sort(np.array(list(set(sample_pops))))
    return sample_pops


def get_pop_diversity(ts, u=1e-8):
    sample_pops = get_sample_populations(ts)
    pi = np.zeros(len(sample_pops))
    for i, pop in enumerate(sample_pops):
        sample = ts.samples(population=pop)
        pi[i] = ts.diversity(mode="branch", sample_sets=sample)
    pi *= u
    return pi, sample_pops


def get_pop_divergence(ts, u=1e-8):
    sample_pops = get_sample_populations(ts)
    div = np.zeros((len(sample_pops), len(sample_pops)))
    for i, pop0 in enumerate(sample_pops):
        sample0 = ts.samples(population=pop0)
        for j, pop1 in enumerate(sample_pops):
            sample1 = ts.samples(population=pop1)
            div[i, j] = ts.divergence(sample_sets=[sample0, sample1],
                                      mode="branch")
    div *= u
    return div, sample_pops



graph = load_yaml(filename="c:/archaic/yamls/super_archaic.yaml")
demog = graph_to_demog(graph)
#ts = coalescent(demog)


goal = 1e8
sizes = [1e3, 1e4, 1e5, 5e5, 1e6, 2.5e6, 5e6, 7.5e6, 1e7, 2e7]


def benchmark(window_lengths, iter=10):
    """
    Record the avg time to run coalescence, mutation and analysis on windows
    of given lengths

    :param window_lengths:
    :return:
    """
    graph = load_yaml(filename="c:/archaic/yamls/super_archaic.yaml")
    demog = graph_to_demog(graph)
    coalescent_times = []
    mutation_times = []
    analysis_times = []
    for length in window_lengths:
        c = []
        m = []
        a = []
        for i in np.arange(iter):
            time0 = time.time()
            ts = sim_ancestry(demog, seq_length=length)
            time1 = time.time()
            c.append(time1 - time0)
            #
            time0 = time.time()
            mts = sim_mutations(ts)
            time1 = time.time()
            m.append(time1 - time0)
            #
            time0 = time.time()
            pi = get_pop_diversity(ts)
            div = get_pop_diversity(ts)
            time1 = time.time()
            a.append(time1 - time0)
        coalescent_times.append(np.mean(c))
        mutation_times.append(np.mean(m))
        analysis_times.append(np.mean(a))
    coalescent_times = np.array(coalescent_times)
    mutation_times = np.array(mutation_times)
    analysis_times = np.array(analysis_times)
    return coalescent_times, mutation_times, analysis_times


def multi_window(demog, window_size=5e5, iter=200):
    time0 = time.time()
    ts_generator = sim_ancestry(demog, seq_length=window_size, reps=iter)
    pi = []
    div = []
    for ts in ts_generator:
        ts_pi, pops = get_pop_diversity(ts)
        pi.append(ts_pi)
        ts_div, pops = get_pop_divergence(ts)
        div.append(ts_div)
    mean_pi = np.mean(pi, axis=0)
    mean_div = np.mean(div, axis=0)
    time1 = time.time()
    print(np.round(time1 - time0, 2))
    return mean_pi, mean_div





















