
"""
For conducting mass inference on a remote server. Outputs a .yaml graph file
"""

import argparse
import demes
from archaic import inference


def get_args():
    # get args
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--boot_fname", required=True)  
    parser.add_argument("-g", "--graph_fname", required=True)
    parser.add_argument("-p", "--param_fname", required=True)
    parser.add_argument("-o", "--out_fname", required=True)
    parser.add_argument("-m", "--max_iter", required=True, type=int)
    parser.add_argument("-v", "--verbosity", type=int, default=0)
    parser.add_argument("-r", "--opt_routine", default="fmin")
    parser.add_argument("-u", "--u", type=float, default=1.35e-8)
    parser.add_argument("-H", "--use_H", type=int, default=1)
    parser.add_argument("-H2", "--use_H2", type=int, default=1)
    return parser.parse_args()


def main():
    #
    sample_names = inference.scan_names(args.graph_fname, args.boot_fname)
    print(sample_names)
    r_bins, data = inference.read_data(args.boot_fname, sample_names)
    graph, etc = inference.optimize(
        args.graph_fname,
        args.param_fname,
        data,
        r_bins,
        args.max_iter,
        verbosity=args.verbosity,
        u=args.u,
        use_H=args.use_H,
        use_H2=args.use_H2
    )
    graph.metadata["fopt"] = - etc[0]
    graph.metadata["iters"] = etc[1]
    graph.metadata["funcalls"] = etc[2]
    graph.metadata["flag"] = etc[3]
    demes.dump(graph, args.out_fname)
    return 0


if __name__ == "__main__":
    args = get_args()
    main()
