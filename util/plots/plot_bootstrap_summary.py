
import argparse
import matplotlib.pyplot as plt
from scipy.stats import norm
from util import file_util
from util import plot_fxns


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("mean_file_name")
    parser.add_argument("std_file_name")
    parser.add_argument("out_dir_name")
    parser.add_argument("-c", "--confidence", type=float, default=0.95)
    parser.add_argument("-y", "--y_lim", type=float, default=3e-6)
    parser.add_argument("-t", "--y_tick", type=float, default=2e-7)
    args = parser.parse_args()
    #
    out_dir_name = args.out_dir_name.rstrip('/')
    means_header, means = file_util.load_arr_as_dict(args.mean_file_name)
    stds_header, stds = file_util.load_arr_as_dict(args.std_file_name)
    factor = norm.ppf(1 - (1 - args.confidence) / 2)
    sample_ids = ["French-1", "Han-1", "Khomani_San-2", "Papuan-2", "Yoruba-1", "Yoruba-3"]
    # group H_2
    H_2_labels = [f"H_2_{x}" for x in sample_ids]
    H_2s = {H_2_label: means[H_2_label] for H_2_label in H_2_labels}
    colors = [plot_fxns.sample_colors[x] for x in sample_ids]
    plot_fxns.plot_curves(
        colors=colors,
        title=f"bootstrap H_2s",
        y_lim=args.y_lim,
        y_tick=args.y_tick,
        **H_2s
    )
    plt.savefig(f"{out_dir_name}/summary.png", dpi=200)
    for x in sample_ids:
        label = f"H_2_{x}"
        plot_fxns.add_curves(
            means[label] - stds[label] * factor,
            means[label] + stds[label] * factor,
            line_style="dotted",
            color=plot_fxns.sample_colors[x]
        )
    plt.savefig(f"{out_dir_name}/summary_confs.png", dpi=200)
    plt.close()
    # sample H_2 and H_2_XY
    for x in sample_ids:
        Hs = {y: means[y] for y in means if x in y}
        colors = []
        line_styles = []
        for y in Hs:
            if "," in y:
                key = y.replace(x, '').replace(',', '').replace('H_2_XY_', '')
                colors.append(plot_fxns.sample_colors[key])
                line_styles.append("dashed")
            else:
                colors.append(plot_fxns.sample_colors[x])
                line_styles.append("solid")
        plot_fxns.plot_curves(
            colors=colors,
            line_styles=line_styles,
            title=f"bootstrap summary; {x}",
            y_lim=args.y_lim,
            y_tick=args.y_tick,
            **Hs
        )
        plt.savefig(f"{out_dir_name}/{x}_summary.png", dpi=200)
        for i, z in enumerate(Hs):
            plot_fxns.add_curves(
                means[z] - stds[z] * factor,
                means[z] + stds[z] * factor,
                line_style="dotted",
                color=colors[i]
            )
        plt.savefig(f"{out_dir_name}/{x}_summary_confs.png", dpi=200)
        plt.close()
