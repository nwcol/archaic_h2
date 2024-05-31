
import argparse
import demes
import matplotlib.pyplot as plt
import numpy as np
from archaic import inference


deme_to_sample = {
    "Altai": "Altai",
    "Chagyrskaya": "Chagyrskaya",
    "Vindija": "Vindija",
    "Denisova": "Denisova",
    "Denisovan": "Denisova",
    "Yoruba": "Yoruba-1",
    "Yoruba1": "Yoruba-1",
    "Yoruba3": "Yoruba-3",
    "KhomaniSan": "Khomani_San-2",
    "Khomani_San": "Khomani_San-2",
    "Papuan": "Papuan-2",
    "French": "French-1",
    "Han": "Han-1"
}
generic_samples = list(deme_to_sample.values())
generic_demes = list(deme_to_sample.keys())


def get_args():

    parser = argparse.ArgumentParser()
    parser.add_argument("-g", "--graph_fname", required=True)
    parser.add_argument("-p", "--param_fname", required=True)
    parser.add_argument("-d", "--boot_archive", required=True)
    parser.add_argument("-o", "--out_prefix", required=True)
    parser.add_argument("-n", "--name_map", type=str, nargs="*", default=[])
    parser.add_argument("-i", "--max_iter", type=int, default=1_000)
    parser.add_argument("-v", "--verbose", type=int, default=10)
    parser.add_argument("-r", "--opt_routine", type=str, default="fmin")
    parser.add_argument("-u", "--u", type=float, default=1.35e-8)
    parser.add_argument("-H", "--use_H", type=int, default=1)
    parser.add_argument("-l", "--log_scale", type=bool, default=False)
    return parser.parse_args()


def get_name_map(all_deme_names):
    # get the names that need to be loaded from data. form samples: demes
    name_map = {}
    arg_map = {x: y for x, y in [m.split(':') for m in args.name_map]}
    for deme in arg_map:
        if deme in all_deme_names:
            name_map[arg_map[deme]] = deme
        else:
            raise ValueError(f"deme {deme} is not present in the graph!")
    for deme in all_deme_names:
        if deme not in name_map.values():
            if deme in deme_to_sample:
                name_map[deme_to_sample[deme]] = deme
    if len(name_map) == 0:
        raise ValueError("sample/deme name configuration selects 0 samples!")
    sample_names = list(name_map.keys())
    deme_names = list(name_map.values())
    return name_map, sample_names, deme_names


def main():

    graph = demes.load(args.graph_fname)
    all_deme_names = [deme.name for deme in graph.demes]
    name_map, sample_names, deme_names = get_name_map(all_deme_names)
    print(name_map)
    r_bins, data = inference.read_data(
        args.boot_archive, sample_names, get_H=args.use_H
    )
    _data = inference.rename_data_samples(data, name_map)
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
    with open(f"{args.out_prefix}log.txt", 'w') as file:
        file.write(str(name_map) + '\n')
        for x in opt:
            file.write(str(x) + '\n')
    out_graph = opt[0]
    demes.dump(out_graph, f"{args.out_prefix}inferred.yaml")
    r_bins, __data = inference.read_data(args.boot_archive, sample_names)
    __data = inference.rename_data_samples(__data, name_map)
    inference.plot(
        out_graph, __data, r_bins, u=args.u, log_scale=args.log_scale,
        use_H=args.use_H
    )
    plt.savefig(f"{args.out_prefix}fit.png", dpi=200, bbox_inches='tight')
    plt.close()


if __name__ == "__main__":
    args = get_args()
    main()
