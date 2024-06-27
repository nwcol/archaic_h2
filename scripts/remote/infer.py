"""
For conducting mass inference on a remote server. Outputs a .yaml graph file
"""


import argparse
import demes
import numpy as np
import moments.Demes.Inference as minf
from archaic import inference


def get_args():
    # get args
    parser = argparse.ArgumentParser()
    parser.add_argument("--cluster_id", default="0")
    parser.add_argument("--process_id", default="0")
    parser.add_argument("-d", "--boot_fname", required=True)
    parser.add_argument("-g", "--graph_fname", required=True)
    parser.add_argument("-p", "--param_fname", required=True)
    parser.add_argument("-o", "--out_fstem", required=True)
    parser.add_argument("-m", "--max_iter", required=True, type=int)
    parser.add_argument("--permute_graph", type=int, default=1)
    parser.add_argument("-v", "--verbosity", type=int, default=0)
    parser.add_argument("-r", "--opt_routine", default="fmin")
    parser.add_argument("-u", "--u", type=float, default=1.35e-8)
    parser.add_argument("-H", "--use_H", type=int, default=1)
    parser.add_argument("-H2", "--use_H2", type=int, default=1)
    return parser.parse_args()


def get_out_fname():
    # get the output filename
    fname = f"{args.out_fstem}_{args.cluster_id}_{args.process_id}.yaml"
    return fname


def get_init_fname():
    # get a filename for the permuted graph
    fname = f"{args.out_fstem}_{args.cluster_id}_{args.process_id}_init.yaml"
    return fname


def log_uniform(lower, upper):
    # sample parameters log-uniformly
    log_lower = np.log10(lower)
    log_upper = np.log10(upper)
    log_draws = np.random.uniform(log_lower, log_upper)
    draws = 10 ** log_draws
    return draws


def permute_graph():
    # uniformly and randomly pick parameter values
    graph = demes.load(args.graph_fname)
    builder = minf._get_demes_dict(args.graph_fname)
    param_dict = minf._get_params_dict(args.param_fname)
    param_names, params0, lower_bounds, upper_bounds = \
        minf._set_up_params_and_bounds(param_dict, builder)
    if np.any(np.isinf(upper_bounds)):
        raise ValueError("all upper bounds must be specified!")
    constraints = minf._set_up_constraints(param_dict, param_names)
    above1 = np.where(lower_bounds >= 1)[0]
    below1 = np.where(lower_bounds < 1)[0]
    n = len(params0)
    satisfied = False
    while not satisfied:
        params = np.zeros(n)
        params[above1] = np.random.uniform(
            lower_bounds[above1], upper_bounds[above1]
        )
        params[below1] = log_uniform(
            lower_bounds[below1], upper_bounds[below1]
        )
        if constraints:
            if np.all(constraints(params) > 0):
                satisfied = True
        else:
            satisfied = True
    builder = minf._update_builder(builder, param_dict, params)
    graph = demes.Graph.fromdict(builder)
    demes.dump(graph, init_fname)


def main():
    #
    args = get_args()
    if args.permute_graph:
        init_fname = get_init_fname()
        permute_graph()
    else:
        init_fname = args.graph_fname
    out_fname = get_out_fname()
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
    main()
