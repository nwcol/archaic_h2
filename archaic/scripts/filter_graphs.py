
import argparse
import demes


def get_args():
    # get args
    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--thresh", required=True, type=int)
    parser.add_argument("-g", "--graph_fnames", nargs='*')
    parser.add_argument("-o", "--out_path", required=True)
    return parser.parse_args()


def main():
    #
    args = get_args()
    out_path = args.out_path.rstrip('/')
    for fname in args.graph_fnames:
        graph = demes.load(fname)
        fopt = int(graph.metadata['opt_info']["fopt"])
        if fopt < args.thresh:
            print(fname, '\t', fopt)
            basename = fname.split('/')[-1]
            out_fname = f"{out_path}/{basename}"
            demes.dump(graph, out_fname)
    return 0


if __name__ == "__main__":
    main()
