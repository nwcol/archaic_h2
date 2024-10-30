"""
Parse mutation rates and .bed coverage files from a .vcf.gz file recording
estimated per-site mutation rates.
"""
import argparse
import gzip
import numpy as np
import numpy.ma as ma
import re
from h2py import util


# This the coefficient for converting roulette 'MR' rates into haploid
# per-generation mutation rates. Taken from
# https://github.com/vseplyarskiy/Roulette/tree/main/adding_mutation_rate
coeff = 1.015e-7 / 2
rate_info = 'MR'


def get_args():
    #
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-i',
        '--in_file',
        type=str,
        required=True,
        help='Input Roulette .vcf.gz file'
    )
    parser.add_argument(
        '-o',
        '--out_file',
        type=str,
        required=True,
        help='Output filename.'
    )
    parser.add_argument(
        '-b',
        '--out_bed_file',
        type=str,
        required=True,
        help='Output .bed filename.'
    )
    parser.add_argument(
        '--verbosity', 
        type=float, 
        default=50000000
    )
    return parser.parse_args()


def read_seq_len(file):
    """
    Obtain the sequence length from the input file header. 
    """
    seq_lens = dict()
    with gzip.open(file, 'rb') as fin:
        for lineb in fin:
            line = lineb.decode()
            if line.startswith('#'):
                if line.startswith('##contig=<ID'):
                    _, _chrom, _length, *__ = re.split('<|>|,', line)
                    chrom_num = _chrom.split('=')[1]
                    length = int(_length.split('=')[1])
                    # we only want autosomes;
                    if chrom_num.strip('chr').isnumeric():
                        seq_lens[chrom_num] = length
                else:
                    pass
            else:
                chrom_num = line.split()[0]
                break
    seq_len = seq_lens[chrom_num]
    if chrom_num.isnumeric(): chrom_num = f'chr{chrom_num}'
    return chrom_num, seq_len


def main():
    #
    args = get_args()
    chrom_num, seq_len = read_seq_len(args.in_file)
    print(util.get_time(), f'parsing rates on {chrom_num}')
    rates = np.zeros(seq_len, dtype=float)
    mask = np.ones(seq_len, dtype=bool)
    i = 0
    with gzip.open(args.in_file, 'rb') as fin:
        for lineb in fin:
            line = lineb.decode()
            if line.startswith('#'):
                continue
            chrom, pos, _, ref, alt, __, ___, info = line.split()
            if i == 0:
                split_info = info.split(';')
                info_names = [x.split('=')[0] for x in split_info]
                rate_idx = info_names.index(rate_info)
            # we decrement by 1 to get the 0-indexed position
            idx = int(pos) - 1
            split_info = info.split(';')
            rates[idx] += float(split_info[rate_idx].split('=')[1])
            mask[idx] = 0   
            i += 1
            if i % args.verbosity == 0:
                print(util.get_time(), f'parsed rate for {i} rows')

    print(util.get_time(), f'parsed rates for ~{i//3} positions from .vcf')
    rates = rates * coeff
    rates[mask] = np.nan
    np.save(args.out_file, rates)
    regions = util.mask_to_regions(mask)
    util.write_bedfile(args.out_bed_file, chrom_num, regions)
    print(util.get_time(), f'wrote mutation rates to {args.out_file}')


if __name__ == '__main__':
    main()
