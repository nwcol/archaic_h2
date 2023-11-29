import matplotlib.pyplot as plt

import matplotlib

import numpy as np

import statistics


if __name__ == "__main__":
    plt.rcParams['figure.dpi'] = 100
    matplotlib.use('Qt5Agg')


def load_stats(file_name):
    statistics = []
    with open(file_name, 'r') as file:
        for line in file:
            if "{" in line:
                statistics.append(eval(line))
    return statistics


def rename_samples(statistics, name_dict):
    """

    :param stats_dict:
    :param name_dict: map old name (key) to new name (val)
    :return:
    """
    for old in name_dict:
         for stat_dict in statistics:
            if old in stat_dict["sample"]:
                new = name_dict[old]
                sample = tuple([x.replace(old, new) for x in stat_dict["sample"]])
                stat_dict['sample'] = sample
    return statistics


def compute_list_F_2(pi_x_list, pi_xy_list):
    """
    Compute F2 from a list of dictionaries for diversity and divergence

    :param pi_x_list:
    :param pi_xy_list:
    :return:
    """
    pairs = [dic["sample"] for dic in pi_xy_list]
    pi_x_dict = {x["sample"]: x["value"] for x in pi_x_list}
    pi_xy_dict = {x["sample"]: x["value"] for x in pi_xy_list}
    F2_list = []
    for pair in pairs:
        pi_12 = pi_xy_dict[pair]
        pi_1 = pi_x_dict[(pair[0],)]
        pi_2 = pi_x_dict[(pair[1],)]
        F2_12 = statistics.compute_F_2(pi_1, pi_2, pi_12)
        dic = {"sample": pair, "statistic": "F_2", "value": F2_12}
        F2_list.append(dic)
    return F2_list


def construct_matrix(two_way_list):
    pi_xy_dict = {x["sample"]: x["value"] for x in two_way_list}
    all_samples = []
    for x in two_way_list:
        all_samples.append(x["sample"][0])
        all_samples.append(x["sample"][1])
    samples = list(set(all_samples))
    index = np.argsort([order[sample] for sample in samples])
    samples = [samples[i] for i in index]
    n_samples = len(samples)
    indices = []
    for i in np.arange(n_samples):
        for j in np.arange(0, i):
            indices.append((i, j))
    sample_pairs = [(samples[i], samples[j]) for i, j in indices]
    matrix = np.zeros((n_samples, n_samples))
    for index, sample_pair in zip(indices, sample_pairs):
        if sample_pair in pi_xy_dict:
            matrix[index] = pi_xy_dict[sample_pair]
        else:
            inverted_pair = (sample_pair[1], sample_pair[0])
            matrix[index] = pi_xy_dict[inverted_pair]
    return samples, matrix


def plot_one_way(stat_list, statistic, y_max=None):
    dics = [dic for dic in stat_list if dic['statistic'] == statistic]
    dics = sort_list(dics)
    stats = [dic["value"] for dic in dics]
    labels = [dic["sample"][0] for dic in dics]
    fig = plt.figure(figsize=(6, 6.5))
    sub = fig.add_subplot(111)
    x = np.arange(len(stats))
    sub.scatter(x, stats, marker='x', color='black')
    sub.set_xticks(x, labels)
    plt.setp(sub.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")
    annotations = [str(np.format_float_scientific(stat, precision=2))
                   for stat in stats]
    for _x, _y, text in zip(x, stats, annotations):
        sub.annotate(text, xy=(_x, _y), xycoords='data', rotation=45,
                     xytext=(1.5, 1.5), textcoords='offset points',
                     fontsize=8)
    sub.set_title(statistic)
    sub.set_xlabel("population")
    sub.set_ylabel(statistic)
    sub.set_ylim(0,)
    if y_max:
        sub.set_ylim(0, y_max)
    plt.tight_layout()


def plot_two_way(stat_list, focal_sample, statistic, y_max=None):
    dics = [dic for dic in stat_list if dic['statistic'] == statistic
            and focal_sample in dic['sample']]
    full_labels = [dic["sample"] for dic in dics]
    labels = []
    for label in full_labels:
        labels.append([x for x in label if x != focal_sample][0])
    dics = sort_list(dics, samples=labels)
    full_labels = [dic["sample"] for dic in dics]
    labels = []
    for label in full_labels:
        labels.append([x for x in label if x != focal_sample][0])
    stats = [dic["value"] for dic in dics]
    fig = plt.figure(figsize=(6, 6.5))
    sub = fig.add_subplot(111)
    x = np.arange(len(stats))
    sub.scatter(x, stats, marker='x', color='black')
    sub.set_xticks(x, labels)
    plt.setp(sub.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")
    annotations = [str(np.format_float_scientific(stat, precision=2))
                   for stat in stats]
    for _x, _y, text in zip(x, stats, annotations):
        sub.annotate(text, xy=(_x, _y), xycoords='data', rotation=45,
                     xytext=(1.5, 1.5), textcoords='offset points',
                     fontsize=8)
    sub.set_title(f"{statistic} against {focal_sample}")
    sub.set_xlabel("against population")
    sub.set_ylabel(statistic)
    if y_max:
        sub.set_ylim(0, y_max)
    else:
        sub.set_ylim(0,)
    plt.tight_layout()


def plot_matrix(matrix, labels):
    fig, ax = plt.subplots()
    im = ax.imshow(matrix, cmap='Greens')
    ax.set_xticks(np.arange(len(labels)), labels=labels)
    ax.set_yticks(np.arange(len(labels)), labels=labels)
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")
    for i in range(len(labels)):
        for j in range(len(labels)):
            text = ax.text(j, i, matrix[i, j],
                           ha="center", va="center", color="black")
    fig.tight_layout()
    plt.show()


def sort_list(stat_list, samples=None, index=None):
    if not samples:
        samples = [dic['sample'][0] for dic in stat_list]
    if not index:
        index = np.argsort([order[sample] for sample in samples])
        out_list = [stat_list[i] for i in index]
    else:
        out_list = [stat_list[i] for i in index]
    return out_list


def plot_two_ways(stat_list, statistic, y_max=None):
    samples = list(set([dic["sample"][0] for dic in stat_list]))
    for sample in samples:
        plot_two_way(stat_list, sample, statistic, y_max=y_max)


def estimate_branch(F_2_stat, u=1e-8):
    samples = F_2_stat['sample']
    F_2 = F_2_stat['value']
    branch_length = F_2 / u
    out = {'sample': samples, 'statistic': "F_2_branch_length",
           "value": branch_length}
    return out


def estimate_branches(F_2_list, u=1e-8):
    lengths = []
    for dic in F_2_list:
        lengths.append(estimate_branch(dic))
    return lengths


stats = load_stats("c:/archaic/data/genomes/chr22/all_stats.txt")
name_map = {"Vindija33.19": "Neanderthal Vindija",
            "Chagyrskaya-Phalanx": "Neanderthal Chagyrskaya",
            "AltaiNeandertal": "Neanderthal Altai",
            "Denisova": "Denisova",
            "LP6005442-DNA_B02": "Yoruba-1",
            "SS6004475": "Yoruba-3",
            "LP6005592-DNA_C05": "Khomani_San-2",
            "LP6005441-DNA_A05": "French-1",
            "LP6005441-DNA_D05": "Han-1",
            "LP6005443-DNA_H07": "Papuan-2"}
order = {"Denisova": 0,
         "Neanderthal Vindija": 1,
         "Neanderthal Chagyrskaya": 2,
         "Neanderthal Altai": 3,
         "Yoruba-1": 4,
         "Yoruba-3": 5,
         "Khomani_San-2": 6,
         "French-1": 7,
         "Han-1": 8,
         "Papuan-2": 9}
stats = rename_samples(stats, name_map)
diversities = [x for x in stats if x["statistic"] == "pi_x"]
divergences = [x for x in stats if x["statistic"] == "pi_xy"]
F_2s = compute_list_F_2(diversities, divergences)
