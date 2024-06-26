
import argparse
import demes


def get_args():
    # get args
    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--thresh", required=True, type=int)
    parser.add_argument("-f", "--fnames", nargs='*')
    parser.add_argument("-o", "--out_path", required=True)
    return parser.parse_args()


def main():

    out_path = args.out_path.rstrip('/')
    for fname in args.fnames:
        graph = demes.load(fname)
        fopt = int(graph.metadata["fopt"])
        if fopt > args.thresh:
            print(fname, '\t', fopt)
            basename = fname.split('/')[-1]
            out_fname = f"{out_path}/{basename}"
            demes.dump(graph, out_fname)
    return 0


if __name__ == "__main__":
    args = get_args()
    main()

