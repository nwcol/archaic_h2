
import argparse
import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("archive_name")
    parser.add_argument("out_prefix")
    parser.add_argument("-y1", "--y_max1", type=float, default=4e-6)
    parser.add_argument("-y2", "--y_max2", type=float, default=2e-7)
    parser.add_argument("-c", "--ci", type=float, default=1.96)
    return parser.parse_args()


def plot(y, labels, styles=None, marker=None):
    if not styles:
        styles = ["solid"] * len(labels)
    fig, ax = plt.subplots(figsize=(7, 6), layout="constrained")
    ax.grid(alpha=0.2)
    for i, label in enumerate(labels):
        ax.plot(
            r, y[i], color=colors[i], label=label, marker=marker,
            linestyle=styles[i]
        )
    ax.set_xscale("log")
    ax.set_ylim(0, args.y_max1)
    ax.set_xlabel("r bin")
    ax.set_ylabel("H2")
    ax.legend(fontsize=8)


if __name__ == "__main__":
    args = get_args()

    archive = np.load(args.archive_name)
    sample_names = list(archive["sample_names"])
    sample_pairs = list(archive["sample_pairs"])
    r = archive["r_bins"][1:]
    n_samples = len(sample_names)
    n_pairs = len(sample_pairs)
    n = n_samples + n_pairs
    std_H2 = archive["window_H2"].std(1)
    yerrs = std_H2 * args.ci

    colors = cm.nipy_spectral(np.linspace(0, 0.95, len(sample_names)))

    # summary H2 plot
    plot(archive["H2"], sample_names)
    plt.scatter([1] * n_samples, archive["H"] ** 2, color=colors, marker='x')
    plt.savefig(f"{args.out_prefix}H2_summary.png", dpi=200)
    plt.close()

    # archaic plot
    archaics = ["Altai", "Denisova", "Vindija", "Chagyrskaya"]
    idx = np.searchsorted(sample_names, archaics)
    y_max = np.max(archive["H2"][:, idx]) * 1.1
    plot(archive["H2"][idx], archaics)
    plt.ylim(0, y_max)
    plt.savefig(f"{args.out_prefix}H2_archaics.png", dpi=200)
    plt.close()

    # individual plots
    all_ids = sample_names + sample_pairs
    all_H = np.concatenate([archive["H"], archive["Hxy"]], axis=0)
    all_H2 = np.concatenate([archive["H2"], archive["H2xy"]], axis=0)
    for j, sample_name in enumerate(sample_names):
        idx = [i for i in np.arange(n) if sample_name in all_ids[i]]
        # rearrange to retain id order
        idx.insert(j, idx.pop(0))
        idx = np.array(idx)
        ids = [all_ids[i] for i in idx]
        styles = ["dashed"] * n_samples
        styles[j] = "solid"
        plot(all_H2[idx], ids, styles=styles)
        plt.scatter([1] * n_samples, all_H[idx], color=colors, marker='x')
        plt.savefig(f"{args.out_prefix}H2_{sample_name}.png", dpi=200)
        plt.close()
