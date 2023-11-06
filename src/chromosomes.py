import matplotlib.pyplot as plt

import matplotlib

import numpy as np


if __name__ == "__main__":
    plt.rcParams['figure.dpi'] = 100
    matplotlib.use('Qt5Agg')


def read(file_name):
    with open(file_name, 'r') as infile:
        for line in infile:
            pass
    return line


def parse_coverage(file_name, est_size=52_000_000, start=20):
    raw = np.zeros(est_size)
    with open(file_name, 'r') as infile:
        for i, line in enumerate(infile):
            if i > start:
                raw[int(line.split("\t")[1])] = 1
    breakpoints = parse(raw)
    return breakpoints


def parse(pos):
    breaks = pos[1:] - pos[:-1]
    breakpoints = np.nonzero(breaks)[0]
    return breakpoints


def plot_coverage(breakpoints, minimum=1000):
    """
    width: gives length of covered areas
    left: gives start of covered areas

    :param breakpoints:
    :return:
    """
    fig = plt.figure(figsize=(12, 3))
    sub = fig.add_subplot(111)
    lengths = breakpoints[1:] - breakpoints[:-1]
    big = np.where(lengths > minimum)[0]
    starts = breakpoints[:-1]
    total_length = np.max(breakpoints)
    sub.barh(y=0, width=total_length, color="white", edgecolor="black")
    sub.barh(y=0, width=lengths[big], left=starts[big], color="blue")
    sub.set_ylim(-2, 2)
    sub.set_xlim(-1e6, total_length+1e6)
    sub.set_yticks(ticks=[0], labels=["chr22"])
    sub.set_xlabel("position")
    fig.show()
