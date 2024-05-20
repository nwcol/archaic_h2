
import argparse
import demes
import demesdraw
import matplotlib.pyplot as plt
import numpy as np
from util import inference
from util.inference import read_data_dir
from util.demography import name_map


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("graph_fname")
    parser.add_argument("param_fname")
    parser.add_argument("boot_archive")
    parser.add_argument("out")
    parser.add_argument("-i", "--max_iter", type=int, default=2_000)
    parser.add_argument("-v", "--verbose", type=int, default=5)
    parser.add_argument("-r", "--opt_routine", type=str, default="fmin")
    args = parser.parse_args()

    # map graph populations to sample ids
    inverted_map = dict(zip(name_map.values(), name_map.keys()))
    all_deme_names = [deme.name for deme in demes.load(args.graph_fname).demes]
    sample_ids = [
        inverted_map[name] for name in all_deme_names if name in inverted_map
    ]
    # load stuff up and run the inference
    r_bins = np.load(args.boot_archive)["r_bins"]
    data = inference.read_data(args.boot_archive, sample_ids)
    # revert sample names to deme names
    deme_names = [name_map[name] for name in sample_ids]
    print(sample_ids)
    print(deme_names)
    data = (deme_names, data[1], data[2])
    opt = inference.optimize(
        args.graph_fname,
        args.param_fname,
        data,
        r_bins,
        args.max_iter,
        verbose=args.verbose,
        opt_method=args.opt_routine
    )

    demes.dump(opt[0], f"{args.out}inferred.yaml")
    with open(f"{args.out}log.txt", 'w') as file:
        for x in opt:
            file.write(str(x) + "\n")
    demesdraw.tubes(opt[0])
    plt.savefig(f"{args.out}demes.png", dpi=200)
    plt.close()
    inference.plot(opt[0], data, r_bins, plot_H=True, plot_two_sample=True)
    plt.savefig(f"{args.out}fit.png", dpi=200)
    plt.close()
