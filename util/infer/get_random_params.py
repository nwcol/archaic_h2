

import argparse
import demes
import yaml
from util import demography


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("in_graph_fname")
    parser.add_argument("param_fname")
    parser.add_argument("out_graph_fname")
    args = parser.parse_args()

    graph = demes.load(args.in_graph_fname)
    with open(args.param_fname, 'r') as file:
        param_defines = yaml.safe_load(file)
    out_graph = demography.get_random_params(graph, param_defines)
    demes.dump(out_graph, args.out_graph_fname)
