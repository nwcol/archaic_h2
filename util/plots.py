
#

import matplotlib

import matplotlib.pyplot as plt

import numpy as np

import sys

from util import two_locus


if __name__ == "__main__":
    plt.rcParams['figure.dpi'] = 100
    matplotlib.use('Qt5Agg')


def plot(colors, styles, **kwargs):
    r_mids = two_locus.r_mids
    fig = plt.figure(1)
    ax = plt.subplot(111)
    for i, key in enumerate(kwargs):
        vals = kwargs[key]
        color = colors[i]
        if styles[i] == "x":
            ax.scatter(r_mids, vals, color=color, marker='x', label=key)
        else:
            ax.plot(r_mids, vals, color=color, label=key)
    ax.set_ylim(0, 1e-6)
    ax.set_xscale("log")
    ax.set_ylabel("H_2")
    ax.set_xlabel("r bin")
    ax.legend()
    fig.tight_layout()
    fig.show()
