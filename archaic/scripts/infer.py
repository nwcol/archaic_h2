
"""
You must provide a mapping between sample ids and deme names of the form
-m sample0:deme0 sample1:deme1 ...
"""

import argparse
import demes
import demesdraw
import matplotlib.pyplot as plt
import numpy as np
from util import inference


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("graph_fname")
    parser.add_argument("param_fname")
    parser.add_argument("boot_archive")
    parser.add_argument("out")
    parser.add_argument("-m", "--name_map", type=str, nargs="*")
    parser.add_argument("-i", "--max_iter", type=int, default=2_000)
    parser.add_argument("-v", "--verbose", type=int, default=5)
    parser.add_argument("-r", "--opt_routine", type=str, default="fmin")
    parser.add_argument("-u", "--u", type=float, default=1.35e-8)
    parser.add_argument("-H", "--use_H", type=bool, default=True)
    args = parser.parse_args()

    # map graph populations to sample ids. form deme_name:sample_name
    sample_dict = {x.split(":")[0]: x.split(":")[1] for x in args.name_map}
    sample_demes = list(sample_dict.keys())
    sample_names = list(sample_dict.values())

    all_ids = np.load(args.boot_archive)["sample_names"]
    for sample_name in sample_names:
        if sample_name not in all_ids:
            raise ValueError(f"sample {sample_name} isn't in the given archive!")

    graph = demes.load(args.graph_fname)
    deme_names = [deme.name for deme in graph.demes]
    for sample_deme in sample_demes:
        if sample_deme not in deme_names:
            raise ValueError(f"deme {sample_deme} isn't in the given graph!")

    # load stuff up and run the inference
    r_bins = np.load(args.boot_archive)["r_bins"]
    data = inference.read_data(args.boot_archive, sample_names, get_H=args.use_H)

    print(sample_names)
    print(sample_demes)

    _data = (sample_demes, data[1], data[2])
    opt = inference.optimize(
        args.graph_fname,
        args.param_fname,
        _data,
        r_bins,
        args.max_iter,
        verbose=args.verbose,
        opt_method=args.opt_routine,
        u=args.u,
        use_H=args.use_H
    )
    out_graph = opt[0]
    name_dict = {x: sample_dict[x].replace("-", "").replace("_", "")
                 for x in sample_dict}
    demes.dump(out_graph, f"{args.out}inferred.yaml")
    with open(f"{args.out}log.txt", 'w') as file:
        for x in opt:
            file.write(str(x) + "\n")
    # demesdraw.tubes(out_graph)
    # plt.savefig(f"{args.out}demes.png", dpi=200)
    # plt.close()
    __data = (list(name_dict.values()), data[1], data[2])
    _graph = out_graph.rename_demes(name_dict)
    inference.plot(_graph, __data, r_bins, u=args.u)
    plt.savefig(f"{args.out}fit.png", dpi=200, bbox_inches='tight')
    plt.close()
