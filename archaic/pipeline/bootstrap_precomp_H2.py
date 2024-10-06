"""
uses a separate denominator file. every input file has to have 'chrX_' in its
name
"""
import argparse
import numpy as np

from archaic import parsing, util


def get_args():
    # get args
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--in_fnames", nargs='*', required=True)
    parser.add_argument('-d', '--denominator_fnames', nargs='*', required=True)
    parser.add_argument("-o", "--out_fname", required=True)
    parser.add_argument("-n", "--n_iters", type=int, default=1_000)
    parser.add_argument('--name_map', nargs='*', default=[])
    parser.add_argument('--bin_slice', default=None)
    return parser.parse_args()


def main():

    args = get_args()

    denom_files = dict()
    for fname in args.denominator_fnames:
        if 'chr' not in fname:
            raise ValueError('you must include chr in filename')

        num = int(fname.split('chr')[1].split('_')[0].strip('.npz'))
        assert num in range(23)
        denom_files[num] = np.load(fname)

    print(util.get_time(), f'loaded {len(denom_files)} denominator files')

    in_files = []
    for fname in args.in_fnames:
        if 'chr' not in fname:
            raise ValueError('you must include chr in filename')

        num = int(fname.split('chr')[1].split('_')[0].strip('.npz'))
        assert num in denom_files
        file = dict(np.load(fname))

        for count in ['n_site_pairs']:
            if not np.all(file[count] == 0):
                print(f'overriding pair-counts in data from {fname}')
            assert file[count].shape == denom_files[num][count].shape
            file[count] = denom_files[num][count]

        in_files.append(file)

    print(util.get_time(), f'loaded {len(in_files)} input files')


    if args.bin_slice is None:
        bin_slice = None
    else:
        start, end = args.bin_slice.split('-')
        bin_slice = (int(start), int(end))

    dic = parsing.bootstrap_H2(
        in_files, n_iters=args.n_iters, bin_slice=bin_slice
    )
    if len(args.name_map) > 0:
        ids = dic['ids']
        for mapping in args.name_map:
            old, new = mapping.split(':')
            ids[ids == old] = new
        dic['ids'] = ids
    np.savez(args.out_fname, **dic)
    print(
        util.get_time(),
        f'bootstrapped {len(args.in_fnames)} files'
    )
    return 0


if __name__ == "__main__":
    main()
