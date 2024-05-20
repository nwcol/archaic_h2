
import argparse
import demes
import yaml
from util import demography
from util.demography import name_map, start_times


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("sample_x")
    parser.add_argument("sample_y")
    parser.add_argument("graph_fname")
    parser.add_argument("param_fname")
    # time constants (t_x, t_y) and parameters (T)
    parser.add_argument("-tx", "--t_x", type=float, default=None)
    parser.add_argument("-ty", "--t_y", type=float, default=None)
    parser.add_argument("-T", "--T", type=int, default=1.5e5)
    parser.add_argument("-T_min", "--T_min", type=int, default=1e4)
    parser.add_argument("-T_max", "--T_max", type=int, default=1e6)
    # size parameters (N_A, N_x, N_y)
    parser.add_argument("-N_A", "--N_A", type=int, default=1e4)
    parser.add_argument("-N_A_min", "--N_A_min", type=int, default=100)
    parser.add_argument("-N_A_max", "--N_A_max", type=int, default=1e5)
    parser.add_argument("-N_x", "--N_x", type=int, default=1e4)
    parser.add_argument("-N_y", "--N_y", type=int, default=1e4)
    parser.add_argument("-N_min", "--N_min", type=int, default=100)
    parser.add_argument("-N_max", "--N_max", type=int, default=1e5)
    # migration parameter (m)
    parser.add_argument("-m", "--m", type=float, default=None)
    parser.add_argument("-m_min", "--m_min", type=float, default=1e-8)
    parser.add_argument("-m_max", "--m_max", type=float, default=1e-2)
    # etc
    parser.add_argument("-g", "--generation_time", type=int, default=30)
    args = parser.parse_args()

    # add parameter checks?

    deme_x = name_map[args.sample_x]
    deme_y = name_map[args.sample_y]

    # if end times aren't specified, get them from a dictionary in demography
    if args.t_x:
        t_x = args.t_x
    else:
        t_x = start_times[args.sample_x]
    if args.t_y:
        t_y = args.t_y
    else:
        t_y = start_times[args.sample_y]

    # build graph
    graph = demography.build_pair_demography(
        deme_x,
        deme_y,
        N_A=args.N_A,
        N_x=args.N_x,
        N_y=args.N_y,
        T=args.T,
        t_x=t_x,
        t_y=t_y,
        m=args.m,
        generation_time=args.generation_time
    )

    # build parameters
    param_defines = demography.build_pair_params(
        deme_x,
        deme_y,
        T_min=args.T_min,
        T_max=args.T_max,
        N_A_min=args.N_A_min,
        N_A_max=args.N_A_max,
        N_min=args.N_min,
        N_max=args.N_max,
        m=args.m,
        m_min=args.m_min,
        m_max=args.m_max
    )

    demes.dump(graph, args.graph_fname)
    with open(args.param_fname, 'w') as file:
        yaml.dump(param_defines, file)
