
import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
from archaic import utils


"""
Useful constants
"""


line_styles = [
    "solid",
    "dashed",
    "dotted",
    "dashdot",
    (0, (1, 1)),
    (0, (2, 2)),
    (0, (1, 2, 2, 2)),
    (0, (3, 4, 2, 4))
]


"""
Colors and color maps
"""


def get_gnu_cmap(samples):

    n = len(samples)
    cmap = list(cm.gnuplot(np.linspace(0, 0.95, n)))
    return cmap


"""
Plotting functions dedicated to statistics
"""


def plot_H(ax, H, names, colors, fmts=None, y_errs=None):

    n = len(H)
    if fmts == None:
        fmts = ["."] * n
    else:
        if type(fmts) != list:
            fmts = [fmts] * n
        elif len(fmts) < n:
            fmts = [fmts[0]] * n
    if y_errs == None:
        y_errs = [None] * n
    for i, name in enumerate(names):
        ax.errorbar(i, H[i], color=colors[i], yerr=y_errs[i], fmt=fmts[i])
    ax.set_ylim(0, )
    ax.set_ylabel("$H$")
    ax.set_xticks(np.arange(len(names)), names)
    ax.grid(alpha=0.2)
    return ax


def plot_H2(ax, r, H2, names, colors, styles=None, log_scale=False, y_lim=None):
    # plot several H2 curves. H2 is expected to be shape (n, r) array
    if H2.ndim == 1:
        n = 1
    else:
        n = len(H2)
    if styles == None:
        styles = ["-"] * n
    else:
        if type(styles) != list:
            styles = [styles] * n
        elif len(styles) < n:
            styles = [styles[0]] * n

    for i, name in enumerate(names):
        ax.plot(
            r, H2[i], color=colors[i], linestyle=styles[i], label=name
        )
    ax.set_xscale("log")
    ax.set_ylabel("$H_2$")
    ax.set_xlabel("r")
    ax.grid(alpha=0.2)
    if log_scale:
        ax.set_yscale("log")
    else:
        if y_lim:
            ax.set_ylim(0, y_lim)
        else:
            ax.set_ylim(0, )
    return ax


def plot_H2_err(ax, r, H2, names, colors, fmts=None, y_errs=None, log_scale=False, y_lim=None):
    # plot several H2 curves. H2 is expected to be shape (n, r) array
    if H2.ndim == 1:
        n = 1
    else:
        n = len(H2)
    if fmts == None:
        fmts = ["-"] * n
    else:
        if type(fmts) != list:
            fmts = [fmts] * n
        elif len(fmts) < n:
            fmts = [fmts[0]] * n
    if y_errs == None:
        y_errs = [None] * n

    for i, name in enumerate(names):
        ax.errorbar(
            r, H2[i], yerr=y_errs[i], color=colors[i], fmt=fmts[i], label=name
        )
    ax.set_xscale("log")
    ax.set_ylabel("$H_2$")
    ax.set_xlabel("r")
    ax.grid(alpha=0.2)
    if log_scale:
        ax.set_yscale("log")
    else:
        if y_lim:
            ax.set_ylim(0, y_lim)
        else:
            ax.set_ylim(0, )
    return ax

