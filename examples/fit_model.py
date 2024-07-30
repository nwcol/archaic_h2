"""
Fit a demes model to H2 data. Usage is:
python fit_model.py -d [data filename] -g [graph filename] \
    -p [options filename] -o [output filename]
"""
import argparse
import demes

from archaic import inference, utils
from archaic.spectra import H2Spectrum


def get_args():
    # get args
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--data_fname', required=True)
    parser.add_argument('-g', '--graph_fname', required=True)
    parser.add_argument('-p', '--options_fname', required=True)
    parser.add_argument('-o', '--out_fname', required=True)
    parser.add_argument('-u', '--u', type=float, default=1.35e-8)
    parser.add_argument('--max_iter', type=int, default=1000)
    parser.add_argument('--method', default='NelderMead')
    parser.add_argument('-v', '--verbosity', type=int, default=1)
    return parser.parse_args()


def main():
    #
    args = get_args()

    # load H2 statistics from the .npz bootstrap archive
    data = H2Spectrum.from_bootstrap_file(
        args.data_fname, graph=demes.load(args.graph_fname)
    )
    print(utils.get_time(), f'running inference for demes {data.sample_ids}')

    # run the inference
    inference.fit_H2(
        args.graph_fname,
        args.options_fname,
        data,
        max_iter=args.max_iter,
        method=args.method,
        u=args.u,
        verbosity=args.verbosity,
        use_H=True,
        out_fname=args.out_fname
    )
    return 0


if __name__ == '__main__':
    main()


