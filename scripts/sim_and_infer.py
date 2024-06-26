

import argparse
import demes
import numpy as np
import moments.Demes.Inference as minf
from archaic import inference
from archaic import coalescent
from archaic.scripts import parse_H2


def get_args():
    # get args
    parser = argparse.ArgumentParser()
    parser.add_argument("--cluster_id", default="0")
    parser.add_argument("--process_id", default="0")
    parser.add_argument("-g", "--graph_fname", required=True)
    parser.add_argument("-p", "--param_fname", required=True)
    parser.add_argument("-n", "--n_windows", type=int, required=True)
    parser.add_argument("-L", "--L", type=float, required=True)
    parser.add_argument("-o", "--out_fstem", required=True)
    parser.add_argument("-m", "--max_iter", required=True, type=int)
    parser.add_argument("-s", "--sample_names", nargs='*', required=True)
    parser.add_argument("-v", "--verbosity", type=int, default=0)
    parser.add_argument("--opt_routine", default="fmin")
    parser.add_argument("-u", "--u", type=float, default=1.35e-8)
    parser.add_argument("-r", "--r", type=float, default=1e-8)
    return parser.parse_args()


def get_out_fname():
    # get the output filename 
    fname = f"{args.out_fstem}_{args.cluster_id}_{args.process_id}.yaml"
    return fname


def main():
    #
    for i in range(args.n_windows):
        fname = f"win{i}.vcf"
        coalescent.generic_coalescent(
            args.graph_fname,
            fname,
            args.sample_names,
            args.L,
            r=args.r,
            u=args.u
        )
        parse_H2.parse(
            args.
        )




    sample_names = inference.scan_names(args.graph_fname, args.boot_fname)
    print(sample_names)
    r_bins, data = inference.read_data(args.boot_fname, sample_names)
    graph, etc = inference.optimize(
        init_fname,
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
    graph.metadata["process_id"] = args.process_id
    demes.dump(graph, out_fname)
    return 0


if __name__ == "__main__":
    args = get_args()
    out_fname = get_out_fname()
    main()
