
"""
Functions for various plots
"""

import matplotlib
import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
from util.two_locus import r, r_edges
from util import file_util
from util import sample_sets
from util import maps


if __name__ == "__main__":
    plt.rcParams['figure.dpi'] = 100
    matplotlib.use('Qt5Agg')
    data_path = "/home/nick/Projects/archaic/data"
    stat_path = "/home/nick/Projects/archaic/statistics"

    sample_ids = ["Altai",
                  "Chagyrskaya",
                  "Denisova",
                  "Vindija",
                  "French-1",
                  "Han-1",
                  "Khomani_San-2",
                  "Papuan-2",
                  "Yoruba-1",
                  "Yoruba-3"]

    sample_colors = dict(
        zip(
            sample_ids, cm.nipy_spectral(
                np.linspace(0, 0.9, len(sample_ids))
            )
        )
    )


def plot_r_stat(x, y, ax=None, color="black", label=None, line_style="solid",
                marker=None, scatter=False):
    """
    Plot a vector of statistics as a function of recombination frequency r
    on a log-x axis

    :param x: vector of r-values
    :param y: vector of statistics
    :param ax: optional. default None creates a new axis
    :param color: optional. color for the curve
    :param label: optional. set figure label
    :param line_style: optional. set line style to something other than solid
    :param marker: optional. add a marker style to the plotted curve
    """
    if not ax:
        fig, ax = plt.subplots(figsize=(7, 6), layout="constrained")
        ax.grid(alpha=0.2)
    if scatter:
        ax.scatter(
            x, y, color=color, label=label, marker=marker
        )
    else:
        ax.plot(
            x, y, color=color, linestyle=line_style, label=label, marker=marker
        )
    ax.set_ylim(0, )
    ax.set_xscale("log")
    ax.set_ylabel("H2")
    ax.set_xlabel("r bin")
    return ax


def plot_r_stats(x, dict, colors=None, line_styles=None):

    n_curves = len(dict)
    labels = list(dict.keys())
    if not colors:
        colors = cm.nipy_spectral(np.linspace(0, 0.9, n_curves))
    if not line_styles:
        line_styles = ["solid"] * n_curves
    fig, ax = plt.subplots(figsize=(7, 6), layout="constrained")
    ax.grid(alpha=0.2)
    for i, label in enumerate(labels):
        plot_r_stat(
            x, dict[labels[i]], ax=ax, color=colors[i], label=labels[i],
            line_style=line_styles[i]
        )
    ax.legend(fontsize=7)
    return ax


def plot_all_r_stats(r_bins, site_pairs, het_pairs, two_sample_het_pairs,
                     out_dir_name, sample_colors, dpi=200, y_lim=2e-6,
                     sample_ids=None):
    """
    Make a plot for each individual, with common features!

    :return:
    """
    if not sample_ids:
        sample_ids = list(het_pairs.keys())
    n_samples = len(sample_ids)
    sample_pairs = list(two_sample_het_pairs.keys())
    H2s = {x: het_pairs[x].sum(0) / site_pairs.sum(0) for x in het_pairs}
    H2_2s = {
        x: two_sample_het_pairs[x].sum(0) / site_pairs.sum(0)
        for x in two_sample_het_pairs
    }
    x = r_bins[1:]
    for i, sample_id in enumerate(sample_ids):
        H2_curves = {sample_id: H2s[sample_id]}
        colors = [sample_colors[sample_id]]
        for sample_pair in sample_pairs:
            if sample_id in sample_pair:
                H2_curves[sample_pair] = H2_2s[sample_pair]
                for _id in sample_pair:
                    if _id != sample_id:
                        other_id = _id
                colors.append(sample_colors[other_id])
        line_styles = ["solid"] + ["dotted"] * (n_samples - 1)
        ax = plot_r_stats(x, H2_curves, colors=colors, line_styles=line_styles)
        ax.set_ylim(0, y_lim)
        ax.set_title(f"H2, {sample_id}")
        plt.savefig(f"{out_dir_name}/H2_{sample_id}.png", dpi=dpi)
        plt.close()
    return 0


def plot_conf(x, y, confs, ax=None, color="black", fill_color="blue",
              alpha=0.2, ylim=2.5e-6, label=None):

    if not ax:
        fig, ax = plt.subplots(figsize=(7, 6), layout="constrained")
        ax.set_xscale("log")
        ax.set_ylim(0, ylim)
        ax.grid(alpha=0.3)
        ax.set_ylabel("H_2")
        ax.set_xlabel("r bin")
    ax.plot(x, y, color=color, label=label)
    ax.fill_between(
        x, y - confs, y + confs, color=fill_color, alpha=alpha, linewidth=0.0
    )
    ax.legend(fontsize=6)
    return ax


def plot_afs(afs, ax=None, color="black", label=None, line_style="solid",
             marker=None):
    """
    Plot an allele frequency spectrum. The afs may be given in counts or
    densities.

    :param afs: vector of allele frequency counts OR frequency densities.
        if given in counts, the afs will be normalized to sum to 1.
    :param ax: optional. If None, create a new figure.
    :param color: optional. color for the curve
    :param label: optional. set figure label
    :param line_style: optional. set line style to something other than solid
    :param marker: optional. add a marker style to the plotted curve
    """
    if not ax:
        fig, ax = plt.subplots(figsize=(7.5, 6), layout="constrained")
    if afs.sum() != 1:
        afs = afs / afs.sum()
    x = np.arange(1, len(afs) + 1)
    ax.plot(
        x, afs, color=color, label=label, marker=marker, linestyle=line_style
    )
    ax.set_ylim(0, )
    ax.set_ylabel("frequency")
    ax.set_xlabel("allele frequency")
    ax.set_xticks(x)
    ax.grid(alpha=0.4)
    ax.legend()
    return ax


def plot_dist(bins, data, ax=None, color="black", label=None, x_label="",
              title=None):
    """


    :param bins:
    :param data:
    :param ax:
    :param color:t
    :param label:
    :param x_label:
    :param title:
    :return:
    """
    if not ax:
        fig, ax = plt.subplots(figsize=(6.5, 6), layout="constrained")
    distribution = np.histogram(data, bins=bins)[0]
    distribution = distribution / distribution.sum()
    x = bins[1:]
    ax.plot(x, distribution, color=color, label=label)
    ax.set_ylim(0, )
    ax.set_ylabel("frequency")
    ax.set_xlim(bins[0], bins[-1])
    if not x_label:
        x_label = ""
    ax.set_xlabel(f"{x_label} bin; right edge")
    ax.grid(alpha=0.2)
    ax.legend(fontsize=8)
    if title:
        ax.set_title(title)
    return ax


def plot_along_chr(y, ax=None, x=None, windows=None, color="black", label=None,
                   marker=None, line_style=None, y_label=None):
    """
    Plot a statistic in windows along a chromosome. x-positions are given by
    a list of tuples (chr#, [lower, upper]) or a vector of positions.

    :param x:
    :param y:
    :param ax:
    :param color: optional. color for the curve
    :param label: optional. set figure label
    :param line_style: optional. set line style to something other than solid
    :param marker: optional. add a marker style to the plotted curve
    :param y_label: label for the y-axis
    :return:
    """
    if not ax:
        fig, ax = plt.subplots(figsize=(12, 6), layout="constrained")
    if x is None:
        if windows is None:
            x = np.arange(len(y))
            x_ticks = np.arange(0, max(x) + 5, 5)
            x_tick_labels = x_ticks
            x_label = "window"
        else:
            x = np.array([window[1][1] for window in windows])
            x_ticks = np.arange(0, max(x) + 5e6, 5e6)
            x_tick_labels = np.arange(0, max(x) // 1e6 + 5, 5, dtype=np.int64)
            x_label = "position, Mb"
    else:
        x_ticks = np.arange(0, max(x) + 5e6, 5e6)
        x_tick_labels = np.arange(0, max(x) // 1e6 + 5, 5,  dtype=np.int64)
        x_label = "position, Mb"
    plt.plot(
        x, y, color=color, label=label, marker=marker, linestyle=line_style
    )
    ax.set_ylim(0, )
    ax.set_ylabel(y_label)
    ax.set_xticks(x_ticks, x_tick_labels, rotation=270)
    ax.set_xlim(0, max(x))
    ax.set_xlabel(x_label)
    ax.grid(alpha=0.4)
    ax.legend()
    return ax


def manhattan_plot(windows, y, ax=None, marker='x', colors=None):
    """
    Create a manhattan scatter plot of a statistic y across the genome

    Expected window structure: list of (chrom, [lower, upper])

    :param x:
    :param y:
    :param chroms:
    :param ax:
    :param y_lim:
    :param marker:
    :param colors:
    :return:
    """
    if not colors:
        colors = ["r", "b"]
    chroms = np.array([window[0] for window in windows])
    bounds = np.array([window[1] for window in windows])
    x = np.cumsum(np.diff(bounds))
    n_colors = len(colors)
    chrom_numbers = np.arange(min(chroms), max(chroms) + 1)
    cmap = {i: colors[i % n_colors] for i in chrom_numbers}
    if not ax:
        fig, ax = plt.subplots(figsize=(18, 8), layout="constrained")
    for chrom in range(1, 23):
        mask = chroms == chrom
        ax.plot(x[mask], y[mask], color=cmap[chrom], marker=marker)
    ax.grid(alpha=0.2, axis='y')
    chrom_centers = [x[chroms == i].mean() for i in chrom_numbers]
    ax.set_xticks(chrom_centers, labels=chrom_numbers)
    ax.set_xlim(0, max(x) + x[-1] - x[-2])
    ax.set_xlabel("chromosome")
    ax.set_ylim(0, )
    return ax


def plot_het_pairs(sample_set, sample_id, window, r_bin, ax=None, style="dots",
                   label=None, color="red", x_ax="pos", s=4):
    """
    Plot individual heterozygous site pairs in a triangular field that
    resembles a linkage plot.

    :param sample_set:
    :param sample_id:
    :param window:
    :param r_bin:
    :param ax:
    :param style:
    :param label:
    :param color:
    :param x_ax:
    :param s:
    """
    lower, upper = window
    window_idx = np.nonzero(
        (sample_set.het_sites(sample_id) >= lower)
        & (sample_set.het_sites(sample_id) < upper)
    )[0]
    het_sites = sample_set.het_sites(sample_id)[window_idx]
    het_map = sample_set.het_map(sample_id)[window_idx]
    n_sites = len(window_idx)
    d_bin = map_util.r_to_d(r_bin)
    pairs = []
    n_pairs = 0
    for idx in np.arange(n_sites):
        min_d, max_d = d_bin + het_map[idx]
        idxs = np.nonzero(
            (het_map[idx + 1:] >= min_d) & (het_map[idx + 1:] < max_d)
        )[0]
        idxs += idx
        n_pairs += len(idxs)
        pairs.append(idxs)
    if n_pairs == 0:
        return ax
    if x_ax == "pos":
        x_vec = het_sites
    elif x_ax == "map":
        x_vec = het_map
    else:
        x_vec = het_sites
    x = []
    y = []
    if style == 'vertices':
        for left_idx, site_pairs in enumerate(pairs):
            left_site = x_vec[left_idx]
            for j, right_idx in enumerate(site_pairs):
                right_site = x_vec[right_idx]
                midpoint = np.mean([left_site, right_site])
                distance = right_site - left_site
                x.append([left_site, midpoint, right_site])
                y.append([0, distance / 2, 0])
    elif style == 'dots':
        for left_idx, site_pairs in enumerate(pairs):
            left_site = x_vec[left_idx]
            for j, right_idx in enumerate(site_pairs):
                right_site = x_vec[right_idx]
                midpoint = np.mean([left_site, right_site])
                distance = right_site - left_site
                x.append(midpoint)
                y.append(distance / 2)
    if x_ax == "pos":
        low, high = window
    else:
        low, high = np.min(x), np.max(x)
    if not ax:
        fig, (ax, h_ax) = plt.subplots(
            2, 1, gridspec_kw={'height_ratios': [30, 1]}, figsize=(15, 9),
            sharex=True, layout="constrained"
        )
        # plot the outer boundary
        ax.plot(
            [low, (high + low) / 2, high],
            [0, (high - low) / 2, 0],
            color="black"
        )
        # plot het
        if x_ax == "pos":
            het_locs = het_sites
        else:
            het_locs = het_map
        for loc in het_locs:
            h_ax.plot([loc, loc], [0, 1], color="red", linewidth=0.3)
        h_ax.set_ylim(0, 1)
        if x_ax == "pos":
            h_ax.set_xlabel("position")
        else:
            h_ax.set_xlabel("cM")
        ax.set_xlim(low, high)
        ax.set_ylim(0, (high - low) / 2)
        ax.set_aspect('equal')
        ax.grid(alpha=0.4)
    if style == 'vertices':
        for i in range(len(x)):
            ax.plot(x[i], y[i], color=color, label=label)
    elif style == 'dots':
        ax.scatter(
            x, y, color=color, marker=',', s=s, linewidths=0, label=label
        )
    ax.legend()
    return ax


def plot_het_pair_spectrum(sample_set, sample_id, window, r_edges, s=4,
                           x_ax="pos"):

    n_bins = len(r_edges) - 1
    colors = cm.nipy_spectral(np.linspace(0, 0.9, n_bins))
    ax = plot_het_pairs(
        sample_set, sample_id, window, r_edges[0:2], color=colors[0],
        label=f"r_({r_edges[0]}, {r_edges[1]})", s=s, x_ax=x_ax
    )
    for i in range(1, n_bins):
        plot_het_pairs(
            sample_set, sample_id, window, r_edges[i:i + 2], color=colors[i],
            ax=ax, label=f"r_({r_edges[i]}, {r_edges[i + 1]})", s=s, x_ax=x_ax
        )
    ax.set_title(f"{sample_id} {sample_set.chrom}:{window}")
    return ax


def triangle_window_plot(window_dict, genetic_map, max_r):
    """


    :param window_dict:
    :param genetic_map:
    :return:
    """
    fig, ax = plt.subplots(figsize=(8, 5), layout="constrained")
    colors = cm.nipy_spectral(np.linspace(0.1, 0.9, len(window_dict["windows"])))
    low = 0
    high = genetic_map.last_position
    x = [low, (high - low) / 2, high]
    y = [0, (high - low) / 2, 0]
    ax.plot(x, y, color="black")

    for k, window_id in enumerate(window_dict["windows"]):
        start, stop = window_dict["windows"][window_id]["bounds"]
        limit_right = window_dict["windows"][window_id]["limit_right"]
        stop = min(high, stop)
        middle = (start + stop) / 2
        top = (stop - start) / 2
        x = [start, middle, stop]
        y = [0, top, 0]
        ax.plot(x, y, color=colors[k])
        step = 100_000
        starts = np.arange(start, stop + step, step)
        if limit_right:
            stops = np.array(
                [min(genetic_map.find_interval_end(i, max_r), stop)
                 for i in starts]
            )
        else:
            stops = np.array(
                [genetic_map.find_interval_end(i, max_r) for i in starts]
            )
        x = np.array([starts, stops]).mean(0)
        y = x - starts
        ax.plot(x, y, color=colors[k], linestyle="dotted")
    ax.set_xlim(low, high)
    ax.set_xlabel("position")
    ax.set_ylim(0, (high - low) / 2)
    ax.set_ylabel("position range")
    ax.set_aspect('equal')
    ax.grid(alpha=0.4)
    return ax


"""
Utilities
"""


def setup_ticks(ax, interval=5, minor_interval=None, max_x=None):

    if not max_x:
        max_x = np.ceil(ax.lines[0].get_xydata()[:, 0].max() / 1e6)
    labels = np.arange(0, max_x, interval, dtype=np.int64)
    ticks = labels * 1e6
    ax.set_xticks(ticks, labels=labels, rotation=270)
    if minor_interval:
        minor_ticks = np.arange(0, max_x, minor_interval) * 1e6
        ax.set_xticks(minor_ticks)


"""
Older, less-polished functions
"""


def plot_bed_coverage(res=1e6, title=None, styles=None, colors=None, **beds):
    last_position = max([beds[key].last_position for key in beds])
    n_bins = last_position // res + 1
    bins = np.arange(0, (n_bins + 1) * res, res)
    bins_Mb = bins / 1e6
    fig = plt.figure(figsize=(12, 5))
    ax = fig.add_subplot(111)
    if not colors:
        colors = ["black"] * len(beds)
    if not styles:
        styles = ["solid"] * len(beds)
    for i, key in enumerate(beds):
        positions = beds[key].positions_0
        pos_counts, dump = np.histogram(positions, bins=bins)
        coverage = pos_counts / res
        ax.plot(bins_Mb[1:], coverage, label=key, color=colors[i],
                linestyle=styles[i])
    ax.set_ylim(-0.01, 1.01)
    ax.set_xlim(0, bins_Mb[-1])
    ax.set_xlabel("position, Mb")
    ax.set_ylabel("bin coverage")
    ax.legend()
    #ticks = np.arange(0, bins_Mb[-1], 10, dtype=np.int64)
    #ax.set_xticks(ticks)
    #ax.set_xticks(np.arange(0, bins_Mb[-1] + 1, 1), minor=True)
    ax.grid()
    ax.grid(which="minor", alpha=0.5)
    if title:
        ax.set_title(title)
    fig.tight_layout()
    fig.show()


def plot_exact_mask(colors=None, **beds):

    fig = plt.figure(figsize=(12, 5))
    ax = fig.add_subplot(111)
    if not colors:
        colors = ["black"] * len(beds)
    for i, key in enumerate(beds):
        region_edges = beds[key].parse().flat_regions
        n_regions = len(region_edges)
        x = np.repeat(region_edges, 2)
        y = np.array([0, 1, 1, 0] * (n_regions // 2))
        ax.plot(x, y, color=colors[i], label=key)
    ax.set_ylim(-0.05, 1.05)
    ax.set_xlabel("position, Mb")
    ax.set_ylabel("mask coverage")
    ax.legend()
    setup_ticks(ax)
    ax.grid()
    ax.grid(which="minor", alpha=0.4)
    fig.tight_layout()
    fig.show()


def plot_het_sites(sample_set, res=1e6, sample_ids=None):
    # make a histogram showing n. heterozygous sites for each sample in
    # windows of 'res' along a chromosome
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111)
    n_bins = sample_set.last_position // res + 1
    bins = np.arange(0, (n_bins + 1) * res, res)
    if not sample_ids:
        sample_ids = sample_set.sample_ids
    for sample_id in sample_ids:
        het_sites = sample_set.het_sites(sample_id)
        counts, bins = np.histogram(het_sites, bins=bins)
        plt.plot(bins[1:], counts, color=colors[sample_id], label=sample_id)
    ax.legend()
    ax.set_ylim(0,)
    ax.set_ylabel("n het sites")
    ax.set_xlabel("position, bp")
    fig.tight_layout()


def plot_het(sample_set, res=1e6, title=None, sample_ids=None, ylim=None):
    # make a histogram showing n. heterozygous sites for each sample in
    # windows of 'res' along a chromosome
    fig, (ax0, ax1) = plt.subplots(
        2, 1, gridspec_kw={'height_ratios': [4, 1]}, figsize=(12, 7),
        sharex=True
    )
    n_bins = sample_set.last_position // res + 1
    bins = np.arange(0, (n_bins + 1) * res, res)
    bins_Mb = bins / 1e6
    positions = sample_set.positions
    pos_counts, dump = np.histogram(positions, bins=bins)
    coverage = pos_counts / res
    ax1.plot(bins_Mb[1:], coverage, color="black")
    ax1.set_ylim(0, 1.01)
    ax1.set_xlim(0, bins[-1])
    ax1.set_xlabel("position, Mb")
    ax1.set_ylabel("bin coverage")
    ticks = np.arange(0, bins_Mb[-1] + 10, 10, dtype=np.int64)
    ax1.set_xticks(ticks)
    ax1.set_xticks(np.arange(0, bins_Mb[-1] + 1, 1), minor=True)
    if not sample_ids:
        sample_ids = sample_set.sample_ids
    for sample_id in sample_ids:
        het_sites = sample_set.het_sites(sample_id)
        counts, dump = np.histogram(het_sites, bins=bins)
        ax0.plot(bins_Mb[1:], counts/pos_counts, color=colors[sample_id],
                 label=sample_id)
    ax0.legend()
    if ylim:
        ax0.set_ylim(0, ylim)
    else:
        ax0.set_ylim(0,)
    ax0.set_xlim(0, bins_Mb[-1])
    ax0.set_ylabel("bin H")
    ax0.grid()
    ax1.grid()
    ax0.grid(which="minor", alpha=0.5)
    ax1.grid(which="minor", alpha=0.5)
    if title:
        ax0.set_title(title)
    fig.tight_layout()


def coverage_plot_str(bed):
    positions = bed.get_0_idx_positions()
    approx_max = np.round(bed.last_position, -6) + 1e6
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
