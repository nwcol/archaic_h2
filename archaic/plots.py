
import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
from archaic import utils


"""
Useful constants
"""


_line_styles = [
    "solid",
    "dashed",
    "dotted",
    "dashdot",
    (0, (1, 1)),
    (0, (2, 2)),
    (0, (1, 2, 2, 2)),
    (0, (3, 4, 2, 4))
]


_colors = [
    "red",
    "blue",
    "green"
]


"""
Colors and color maps
"""


def get_gnu_cmap(n):

    cmap = list(cm.gnuplot(np.linspace(0, 0.95, n)))
    return cmap


def get_terrain_cmap(n):

    cmap = list(cm.terrain(np.linspace(0, 0.95, n)))
    return cmap


"""
Plotting graph statistics
"""


def plot_curves(axs, H, H2, r, sample_names, pair_names, color, log_scale,
                label=None):

    shape = axs.shape
    offset = 1
    n = len(sample_names)
    plot_H(axs[0, 0], H[:n], sample_names, color, label=label, title="$H$")
    if len(pair_names) > 0:
        offset += 1
        plot_H(axs[0, 1], H[n:], pair_names, color, title="$H_{xy}$")
    for i in range(len(sample_names)):
        idx = np.unravel_index(i + offset, shape)
        title = f"$H_2$:{sample_names[i]}"
        plot_H2(axs[idx], r, H2[:, i], color, log_scale=log_scale, title=title)
    offset += n
    for i in range(len(pair_names)):
        idx = np.unravel_index(i + offset, shape)
        title = f"$H_{{2,xy}}$:{pair_names[i]}"
        plot_H2(axs[idx], r, H2[:, i], color, log_scale=log_scale, title=title)
    return 0


def plot_error_points(axs, H, H_err, H2, H2_err, r, sample_names, pair_names,
                      color, log_scale, label=None):

    shape = axs.shape
    offset = 1
    n = len(sample_names)
    plot_H_err(axs[0, 0], H[:n], H_err[:n], sample_names, color, label=label, title="$H$")
    if len(pair_names) > 0:
        offset += 1
        plot_H_err(axs[0, 1], H[n:], H_err[n:], pair_names, color, title="$H_{xy}$")
    for i in range(len(sample_names)):
        idx = np.unravel_index(i + offset, shape)
        title = f"$H_2$:{sample_names[i]}"
        plot_H2_err(axs[idx], r, H2[:, i], H2_err[:, i], color, log_scale=log_scale, title=title)
    offset += len(sample_names)
    for i in range(len(pair_names)):
        idx = np.unravel_index(i + offset, shape)
        title = f"$H_{{2,xy}}$:{pair_names[i]}"
        plot_H2_err(axs[idx], r, H2[:, i], H2_err[:, i], color, log_scale=log_scale, title=title)
    return 0


"""
Plotting functions dedicated to statistics
"""


def plot_H(ax, H, names, color, label=None, title=None):

    for i, H in enumerate(H):
        if i >= 1:
            label = None
        ax.scatter(i, H, color=color, marker="_", label=label)
    ax.set_xticks(np.arange(len(names)), names)
    ax.grid(alpha=0.2)
    if title:
        ax.set_title(title)


def plot_H_err(ax, H, H_err, names, color, label=None, title=None):

    for i, H in enumerate(H):
        if i >= 1:
            label = None
        ax.errorbar(i, H, yerr=H_err[i], color=color, fmt='.', label=label)
    ax.set_xticks(np.arange(len(names)), names)
    ax.grid(alpha=0.2)
    if title:
        ax.set_title(title)


def _plot_H_err(ax, H, H_err, E_H, names, colors, E_colors, title=None):

    abbrev_names = []
    for i, name in enumerate(names):
        if type(name) == str:
            name = name[:3]
        else:
            name = f"{name[0][:3]}-{name[1][:3]}"
        abbrev_names.append(name)
        ax.errorbar(i, H[i], yerr=H_err[i], color=colors[i], fmt='.')
        ax.scatter(i, E_H[i], color=E_colors[i], marker='+')
    ax.set_ylim(0, )
    ax.set_xticks(np.arange(len(names)), abbrev_names)
    ax.grid(alpha=0.2)
    if title:
        ax.set_title(title)
    return ax


def plot_H2(ax, r, H2, color, log_scale=False, title=None):
    # for plotting expectations
    ax.plot(r, H2, color=color)
    ax.set_xscale("log")
    ax.autoscale()
    ax.grid(alpha=0.2)
    if log_scale:
        ax.set_yscale("log")
    if title:
        ax.set_title(title)
    return ax


def plot_H2_err(ax, r, H2, H2_err, color, log_scale=False, title=None):
    # for plotting empirical values
    ax.errorbar(r, H2, yerr=H2_err, color=color, fmt=".", capsize=0)
    ax.set_xscale("log")
    ax.grid(alpha=0.2)
    if log_scale:
        ax.set_yscale("log")
    else:
        ax.set_ylim(0, )
    if title:
        ax.set_title(title)
    return ax


def _plot_H2_err(ax, r, H2, H2_err, E_H2, color, E_color, log_scale=False,
                title=None):

    ax.errorbar(
        r, H2, yerr=H2_err, color=color, fmt=".", capsize=0
    )
    ax.plot(r, E_H2, color=E_color)
    ax.set_xscale("log")
    if log_scale:
        ax.set_yscale("log")
    ax.grid(alpha=0.2)
    if title:
        ax.set_title(title)
    return ax


def _plot_H2_err(ax, r, H2, names, colors, fmts=None, y_errs=None, log_scale=False, y_lim=None, title=None):
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
    if title:
        ax.set_title(title)
    return ax

