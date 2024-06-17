
"""
Given graph and parameter files, construct and save a new graph by randomly and
uniformly sampling values from  parameter bounds.
"""

import argparse
import demes
import numpy as np
import moments.Demes.Inference as minf


def get_args():
    # get args
    parser = argparse.ArgumentParser()
    parser.add_argument("-g", "--graph_fname", required=True)
    parser.add_argument("-p", "--param_fname", required=True)
    parser.add_argument("-o", "--out_fname", required=True)
    return parser.parse_args()


def main():
    # uniformly sample parameters until constraints are satisfied 
    graph = demes.load(args.graph_fname)
    builder = minf._get_demes_dict(args.graph_fname)
    param_dict = minf._get_params_dict(args.param_fname)
    param_names, params0, lower_bounds, upper_bounds = \
        minf._set_up_params_and_bounds(param_dict, builder)
    if np.any(np.isinf(upper_bounds)):
        raise ValueError("all upper bounds must be specified!")
    constraints = minf._set_up_constraints(param_dict, param_names)
    satisfied = False
    while not satisfied:
        params1 = np.random.uniform(lower_bounds, upper_bounds)
        if constraints:
            if np.all(constraints(params1) > 0):
                satisfied = True
        else:
            satisfied = True
    builder = minf._update_builder(builder, param_dict, params1)
    graph = demes.Graph.fromdict(builder)
    demes.dump(graph, args.out_fname)
    return 0


if __name__ == "__main__":
    args = get_args()
    main()

