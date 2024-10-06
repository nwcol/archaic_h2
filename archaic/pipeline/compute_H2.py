
import argparse
import numpy as np

from archaic import util, parsing


def get_args():
    #
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--in_fnames', required=True, nargs='*')
    parser.add_argument('-d', '--denominator_fnames', nargs='*', default=None)
    parser.add_argument('-o', '--out_fname', required=True)
    parser.add_argument('--mode', default='between')
    return parser.parse_args()


def main():
    #
    args = get_args()

    if args.mode not in ['within', 'between']:
        raise ValueError('invalid mode supplied')

    if args.denominator_fnames is None:
        files = []
        for fname in args.fnames:
            dic = dict(np.load(fname)) 
            if args.mode == 'within':
                files.append(dic)
            else:
                for key in ['n_sites', 'n_site_pairs', 'H_counts', 'H2_counts']:
                    dic[key] = dic[key].sum(0)[np.newaxis]
                files.append(dic)

    else:
        denom_files = dict()
        for fname in args.denominator_fnames:
            if 'chr' not in fname:
                raise ValueError('you must include chr in filename')

            num = int(fname.split('chr')[1].split('_')[0])
            assert num in range(23)
            if args.mode == 'within':
                denom_files[num] = dict(np.load(fname))
            else:
                dic = dict(np.load(fname))
                dic['n_site_pairs'] = dic['n_site_pairs'].sum(0)[np.newaxis]
                dic['n_sites'] = dic['n_sites'].sum()
                denom_files[num] = dic

        print(util.get_time(), f'loaded {len(denom_files)} denominator files')

        files = []
        for fname in args.in_fnames:
            if 'chr' not in fname:
                raise ValueError('you must include chr in filename')

            num = int(fname.split('chr')[1].split('_')[0])
            assert num in denom_files
            file = dict(np.load(fname))

            if args.mode == 'within':
                pass
            else:
                file = dict(np.load(fname))
                # site/pair counts already summed above
                file['H2_counts'] = file['H2_counts'].sum(0)[np.newaxis]
                file['H_counts'] = file['H_counts'].sum(0) 
                for count in ['n_site_pairs', 'n_sites']:
                    # assert np.all(file[count] == 0)
                    # assert file[count].shape == denom_files[num][count].shape
                    file[count] = denom_files[num][count]
            files.append(file)

    print(util.get_time(), f'loaded {len(files)} input files')

    dic = parsing.get_mean_H2(*files)
    np.savez(args.out_fname, **dic)


if __name__ == '__main__':
    main()
