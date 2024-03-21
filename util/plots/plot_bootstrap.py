
import argparse
import matplotlib.pyplot as plt
from scipy.stats import norm
from util import file_util
from util import plotting


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
    # individual plots
    for x in means:
        mean = means[x]
        lower = mean - stds[x] * factor
        upper = mean + stds[x] * factor
        plot_fxns.plot_curves(
            line_styles=["solid", "dotted", "dotted"],
            title=f"bootstrap {x}",
            y_lim=args.y_lim,
            y_tick=args.y_tick,
            **{
                "mean": mean,
                f"ci {args.confidence} lower": lower,
                f"ci {args.confidence} upper": upper
            }
        )
        plt.savefig(f"{out_dir_name}/{x}.png", dpi=200)
        plt.close()

