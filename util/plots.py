
#

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import sys
from util import two_locus
from util import bed_util


if __name__ == "__main__":
    plt.rcParams['figure.dpi'] = 100
    matplotlib.use('Qt5Agg')


def get_coverage_str(bed):
    positions = bed.get_0_idx_positions()
    approx_max = np.round(bed.max_pos, -6) + 1e6
    out = []
    icons = {
        0.01: "~",
        0.1: "-",
        0.25: "=",
        0.5: "*",
        0.75: "x",
        0.9: "X",
        1: "$"
    }
    key = ", ".join([f"({icons[key]}) cover < {key * 100}%" for key in icons])
    print(key)
    for i in np.arange(0, approx_max, 1e6, dtype=np.int64):
        n_positions = np.searchsorted(positions, i + 1e6) - \
                      np.searchsorted(positions, i)
        coverage = n_positions / 1e6
        for lim in icons:
            if coverage < lim:
                out.append(icons[lim])
                break
    out = "".join(out)
    return out


def plot(**kwargs):

    r_mids = two_locus.r_bin_mids
    fig = plt.figure(1)
    ax = plt.subplot(111)
    for i, key in enumerate(kwargs):
        vals = kwargs[key]
        ax.plot(r_mids, vals, marker='x', label=key)
    ax.set_ylim(0, 1e-6)
    ax.set_xscale("log")
    ax.set_ylabel("H_2")
    ax.set_xlabel("r bin")
    ax.legend()
    fig.tight_layout()
    fig.show()
