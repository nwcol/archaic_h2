
#

import matplotlib
import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
from util.two_locus import r
from util import file_util


if __name__ == "__main__":
    plt.rcParams['figure.dpi'] = 100
    matplotlib.use('Qt5Agg')


def plot_curves(line_styles=None, colors=None, y_lim=2.4e-6, y_tick=2e-7,
                title=None, **kwargs):
    """
    Plot an arbitrary number of curves on a log-scaled-x figure
    """
    if not colors:
        colors = ["black"] * len(kwargs)
    if not line_styles:
        line_styles = ["solid"] * len(kwargs)
    fig = plt.figure(figsize=(7, 6))
    ax = plt.subplot(111)
    for i, key in enumerate(kwargs):
        ax.plot(r,
                kwargs[key],
                color=colors[i],
                linestyle=line_styles[i],
                label=key)
    ax.set_yticks(np.arange(0, y_lim + y_tick, y_tick))
    ax.set_ylim(0, y_lim)
    ax.set_xlim(r.min(), r.max())
    ax.set_xscale("log")
    ax.set_ylabel("H_2")
    ax.set_xlabel("r bin")
    ax.grid(alpha=0.4)
    ax.legend(fontsize=7)
    if title:
        ax.set_title(title)
    fig.tight_layout()
    fig.show()


def add_curves(*args, line_style="solid", color="black"):

    for y in args:
        plt.plot(r, y, linestyle=line_style, color=color)


def manhattan_plot(stat_vec, x_vec, chr_vec, y_lim, title):
    # make a manhattan plot of a statistic along the whole genome.
    colors = ["r", "b"]
    n_colors = len(colors)
    cmap = {}
    for i in np.arange(1, 23):
        cmap[i] = colors[i % n_colors]
    colors = [cmap[x] for x in chr_vec]
    chr_centers = [x_vec[chr_vec == i].mean() for i in np.arange(1, 23)]
    fig = plt.figure(figsize=(15, 5))
    ax = fig.add_subplot(111)
    ax.scatter(x_vec, stat_vec, marker="x", color=colors)
    ax.grid(alpha=0.5, axis='y')
    ax.set_xticks(chr_centers, labels=np.arange(1, 23))
    ax.set_xlim(0, max(x_vec) + 1)
    ax.set_xlabel("chromosome")
    ax.set_ylabel("H_2")
    ax.set_ylim(0, y_lim)
    outlier_mask = stat_vec > y_lim
    outlier_colors = [colors[i] for i in np.nonzero(outlier_mask)[0]]
    ax.scatter(
        x_vec[outlier_mask], [y_lim] * np.sum(outlier_mask), marker='^',
        color=outlier_colors
    )
    for i in np.nonzero(outlier_mask)[0]:
        ax.annotate(
            f"{stat_vec[i]:.2e}", (x_vec[i], y_lim-8e-6), rotation=-90
        )
    ax.set_title(title)
    fig.tight_layout()
















"""
PLOTTING H2 FOR EACH INDIVIDUAL UNDER SEVERAL MASKS
sample_id = ("Khomani_San-2")
fig = plt.figure(figsize=(7, 6))
ax = plt.subplot(111)
ax.plot(r, H2[sample_id], color="blue", label="unfiltered")
ax.plot(r, exonless_H2[sample_id], color="red", label="exon mask")
ax.plot(r, flank_5kb_H2[sample_id], color="black", label="5kb exon flank mask")
ax.plot(r, flank_10kb_H2[sample_id], color="black", label="10kb exon flank mask", linestyle="dashed")
ax.plot(r, flank_20kb_H2[sample_id], color="black", label="20kb exon flank mask", linestyle="dashdot")
ax.plot(r, flank_40kb_H2[sample_id], color="black", label="40kb exon flank mask", linestyle="dotted")
ax.set_yticks(np.arange(0, 2.6e-6, 2e-7))
ax.set_ylim(0, 2.4e-6)
ax.set_xlim(1e-7, 0.1)
ax.grid(alpha=0.4)
ax.legend()
ax.set_xscale("log")
ax.set_title(f"exon flank masking; {sample_id}")
ax.set_ylabel("H_2")
ax.set_xlabel("r bin")
fig.tight_layout()
plt.savefig(f"{stat_path}/exonless/figures/H2_comparison_{sample_id}.png",dpi=200)
plt.close()



PLOTTING MASK COVERAGES
for i in range(1, 23):
    og = bed_util.Bed.read_chr(i)
    no_ex = bed_util.Bed.read_bed(
        f"{data_path}/masks_etc/no_exons/chr{i}_no_exons.bed")
    flank5 = bed_util.Bed.read_bed(
        f"{data_path}/masks_etc/flank_5kb/chr{i}_flank_5kb.bed")
    flank10 = bed_util.Bed.read_bed(
        f"{data_path}/masks_etc/flank_10kb/chr{i}_flank_10kb.bed")
    flank20 = bed_util.Bed.read_bed(
        f"{data_path}/masks_etc/flank_20kb/chr{i}_flank_20kb.bed")
    flank40 = bed_util.Bed.read_bed(
        f"{data_path}/masks_etc/flank_40kb/chr{i}_flank_40kb.bed")
    plot_bed_coverage(res=5e5,
                      styles=["solid", "solid", "solid", "dashed", "dashdot",
                              "dotted"],
                      colors=["blue", "red", "black", "black", "black",
                              "black"],
                      title=f"exon mask coverages on chr${i}; 500kb bins",
                      **{"unfiltered": og, "exon mask": no_ex,
                         "5kb exon flank mask": flank5,
                         "10kb exon flank mask": flank10,
                         "20kb exon flank mask": flank20,
                         "40kb exon flank mask": flank40})
    plt.savefig(f"{stat_path}/no_exons/figures/coverage/chr{i}_coverage.png", dpi=200)
    plt.close()
"""


def plot(**kwargs):

    fig = plt.figure(figsize=(8, 6))
    ax = plt.subplot(111)
    for i, key in enumerate(kwargs):
        ax.plot(r, kwargs[key], label=key, color=colors[key])
    ax.set_ylim(0, 2e-6)
    ax.set_xlim(1e-7, 0.1)
    ax.set_xscale("log")
    ax.set_ylabel("H_2")
    ax.set_xlabel("r bin")
    ax.grid(which="major", alpha=0.5)
    ax.legend()
    fig.tight_layout()
    fig.show()


def plot_bootstrap(bootstrap_dict, *args, colors=None, line_styles=None,
                   y_lim=2e-6, y_tick=2e-7, title=None):
    # plot the mean plus other selected stats
    if not colors:
        colors = ["black"] * len(args)
    if not line_styles:
        line_styles = ["solid"] * len(args)
    fig = plt.figure(figsize=(8, 6))
    ax = plt.subplot(111)
    ax.plot(r, bootstrap_dict["mean"], color="black", label="mean")
    for i, arg in enumerate(args):
        ax.plot(r, bootstrap_dict[arg], color=colors[i],
                linestyle=line_styles[i], label=arg)
    ax.set_yticks(np.arange(0, y_lim + y_tick, y_tick))
    ax.set_ylim(0, y_lim)
    ax.set_xlim(r.min(), r.max())
    ax.set_xscale("log")
    ax.set_ylabel("H_2")
    ax.set_xlabel("r bin")
    ax.grid(alpha=0.4)
    ax.legend()
    if title:
        ax.set_title(title)
    fig.tight_layout()
    fig.show()


def plot_bootstraps(bootstrap_dicts, *args, line_styles=None,
                   y_lim=2e-6, y_tick=2e-7, title=None):
    # plot the mean plus other selected stats
    if not line_styles:
        line_styles = ["solid"] * len(args)
    fig = plt.figure(figsize=(8, 6))
    ax = plt.subplot(111)
    for key in bootstrap_dicts:
        bootstrap_dict = bootstrap_dicts[key]
        ax.plot(r, bootstrap_dict["mean"], color=colors[key], label=key)
        for i, arg in enumerate(args):
            ax.plot(r, bootstrap_dict[arg], color=colors[key],
                    linestyle=line_styles[i])
    ax.set_yticks(np.arange(0, y_lim + y_tick, y_tick))
    ax.set_ylim(0, y_lim)
    ax.set_xlim(r.min(), r.max())
    ax.set_xscale("log")
    ax.set_ylabel("H_2")
    ax.set_xlabel("r bin")
    ax.grid(alpha=0.4)
    ax.legend()
    if title:
        ax.set_title(title)
    fig.tight_layout()
    fig.show()
























# look over this

def bootstrap_two_locus(pair_counts, arr, sample_size, n_resamplings):
    """
    Bootstrap over an array of window statistics

    :param pair_counts:
    :param arr: arr of stats
    :param sample_size:
    :param n_resamplings:
    :return:
    """
    n_rows, n_cols = np.shape(arr)
    out_arr = np.zeros((n_resamplings, n_cols))

    for i in np.arange(n_resamplings):
        sample_idx = np.random.choice(np.arange(n_rows), size=sample_size,
                                      replace=False)
        sum_statistic = np.sum(arr[sample_idx], axis=0)
        sum_pair_counts = np.sum(pair_counts[sample_idx], axis=0)
        bootstrap_stat = sum_statistic / sum_pair_counts
        out_arr[i] = bootstrap_stat

    out_arr = np.sort(out_arr, axis=0)
    idx_c05 = int(0.05 * n_resamplings) - 1
    idx_c95 = int(0.95 * n_resamplings) - 1
    statistics = {
        "mean": np.mean(out_arr, axis=0),
        "median": np.median(out_arr, axis=0),
        "std": np.std(out_arr, axis=0),
        "minimum": out_arr[0],
        "maximum": out_arr[-1],
        "c05": out_arr[idx_c05],
        "c95": out_arr[idx_c95]
    }
    return statistics


def get_coverage_str(bed):
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


def plot_b00ootstraps(extra_stats, colors=None, **bootstrap_dicts):
    # stats: list
    if not colors:
        colors = cm.hsv(np.linspace(0, 0.9, len(bootstrap_dicts)))
    fig = plt.figure(1)
    ax = plt.subplot(111)
    for i, sample_id in enumerate(bootstrap_dicts):
        bootstrap_dict = bootstrap_dicts[sample_id]
        ax.plot(r, bootstrap_dict["mean"], marker='x', color=colors[i],
                label=sample_id, linewidth=2)
        for stat in extra_stats:
            ax.plot(r, bootstrap_dict[stat], color=colors[i], linestyle=":")
    ax.set_ylim(0, )
    ax.set_xscale("log")
    ax.set_ylabel("H_2")
    ax.set_xlabel("r bin")
    ax.legend()
    fig.tight_layout()
    fig.show()


def hist_het_sites(sample_set, res=1e6, sample_ids=None):
    # make a histogram showing n. heterozygous sites for each sample in
    # windows of 'res' along a chromosome
    fig = plt.figure(1)
    ax = fig.add_subplot(111)
    n_bins = int(sample_set.last_position // res) + 1
    if not sample_ids:
        sample_ids = sample_set.sample_ids
    for sample_id in sample_ids:
        het_sites = sample_set.het_sites(sample_id)
        plt.hist(het_sites, bins=n_bins, alpha=0.5, color=colors[sample_id],
                 label=sample_id)
    ax.legend()
    ax.set_ylabel("n het sites")
    ax.set_xlabel("position, bp")
    fig.tight_layout()


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


def plot_bed_coverage(res=1e6, title=None, styles=None, colors=None, **beds):

    last_position = max([beds[key].last_position for key in beds])
    n_bins = last_position // res + 1
    bins = np.arange(0, (n_bins + 1) * res, res)
    bins_Mb = bins / 1e6
    fig = plt.figure(figsize=(12, 5))
    ax = fig.add_subplot(111)
    if not colors:
        styles = ["black"] * len(beds)

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
    ticks = np.arange(0, bins_Mb[-1], 10, dtype=np.int64)
    ax.set_xticks(ticks)
    ax.set_xticks(np.arange(0, bins_Mb[-1] + 1, 1), minor=True)
    ax.grid()
    ax.grid(which="minor", alpha=0.5)
    if title:
        ax.set_title(title)
    fig.tight_layout()
    fig.show()


def compare_window_H2(sample_id, chrom, rows, old, new, label, title=None):

    fig = plt.figure(figsize=(8, 7))
    ax = fig.add_subplot(111)

    sample_H2_old = old[sample_id]
    sample_H2_new = new[sample_id]

    for row_idx in [x for x in rows if f"chr{chrom}_" in rows[x]]:
        ax.plot(r, sample_H2_old[row_idx], label=rows[row_idx])
        ax.scatter(r, sample_H2_new[row_idx], marker='x',
                    label=f"{label}_{rows[row_idx]}")
    ax.set_ylim(0, 10e-6)
    ax.set_xlim(7e-8, 0.2)
    ax.set_ylabel("window H2")
    ax.set_xlabel("r bin")
    ax.set_xscale("log")
    ax.legend(fontsize=8)
    ax.grid()
    if title:
        ax.set_title(title)
    fig.show()







































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
        sample_ids, cm.nipy_spectral(np.linspace(0, 0.9, 10))
    )
)
