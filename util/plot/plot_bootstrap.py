
import argparse
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
from util.inference import read_data


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("archive_name")
    parser.add_argument("out_prefix")
    parser.add_argument("-y1", "--y_max1", type=float, default=4e-6)
    parser.add_argument("-y2", "--y_max2", type=float, default=2e-7)
    parser.add_argument("-c", "--ci", type=float, default=1.96)
    return parser.parse_args()


def plot(y, err, labels, styles=None):
    if not styles:
        styles = ["solid"] * len(labels)
    fig, ax = plt.subplots(figsize=(7, 6), layout="constrained")
    ax.grid(alpha=0.2)
    for i, label in enumerate(labels):
        ax.errorbar(
            r, y[:, i], yerr=err[:, i], color=colors[i], label=label,
            capsize=2, linestyle=styles[i]
        )
    ax.set_xscale("log")
    ax.set_ylim(0, args.y_max1)
    ax.set_xlabel("r bin")
    ax.set_ylabel("H2")
    ax.legend(fontsize=8)


if __name__ == "__main__":
    args = get_args()
    archive = np.load(args.archive_name)
    sample_ids = list(archive["sample_ids"])
    sample_pairs = list(archive["sample_pairs"])
    r = archive["r_bins"][1:]
    sample_ids, means, covs = read_data(args.archive_name, sample_ids)
    means = means[:-1]  # remove H
    covs = covs[:-1]
    n_samples = len(sample_ids)
    n_pairs = len(sample_pairs)
    n = n_samples + n_pairs
    stds = np.sqrt(covs[:, np.arange(n), np.arange(n)])
    yerrs = stds * args.ci

    colors = cm.nipy_spectral(np.linspace(0, 0.95, len(sample_ids)))

    # summary H2 plot
    plot(means[:, :n_samples], yerrs[:, :n_samples], sample_ids)
    plt.savefig(f"{args.out_prefix}H2_summary.png", dpi=200)
    plt.close()

    # archaic plot
    archaics = ["Altai", "Denisova", "Vindija", "Chagyrskaya"]
    idx = np.searchsorted(sample_ids, archaics)
    y_max = np.max(means[:, idx]) * 1.1
    plot(means[:, idx], yerrs[:, idx], archaics)
    plt.ylim(0, y_max)
    plt.savefig(f"{args.out_prefix}H2_archaics.png", dpi=200)
    plt.close()

    # individual plots
    all_ids = sample_ids + sample_pairs
    for j, sample_id in enumerate(sample_ids):
        idx = [i for i in np.arange(n) if sample_id in all_ids[i]]
        # rearrange to retain id order
        idx.insert(j, idx.pop(0))
        idx = np.array(idx)
        ids = [all_ids[i] for i in idx]
        styles = ["dashed"] * n_samples
        styles[j] = "solid"
        plot(means[:, idx], yerrs[:, idx], ids, styles=styles)
        plt.savefig(f"{args.out_prefix}H2_{sample_id}.png", dpi=200)
        plt.close()


