def read_vcf_rates(
    fname,
    rate_tag='MR',
    verbosity=1e6
):
    # reads mutation rates from a Roulette .vcf.gz file
    verbosity *= 3
    pos_idx = 1
    info_idx = 7
    rate_idx = None
    positions = []
    rates = []
    last_position = None
    i = 0
    with gzip.open(fname, 'rb') as file:
        for line_b in file:
            line = line_b.decode()
            if line[0] == '#':
                continue

            if i == 0:
                _fields = line.strip('\n').split('\t')
                _info = _fields[info_idx].split(';')
                names = [x.split('=')[0] for x in _info]
                if rate_tag not in names:
                    raise ValueError(f'tag {rate_tag} not present in info!')
                rate_idx = names.index(rate_tag)

            fields = line.strip('\n').split('\t')
            position = int(fields[pos_idx])
            info = fields[info_idx].split(';')
            rate = float(info[rate_idx].split('=')[1])

            if position == last_position:
                rates[-1] += rate
            else:
                positions.append(position)
                rates.append(rate)

            last_position = position
            i += 1
            if i % verbosity == 0:
                if i > 0:
                    n = i // 3
                    print(get_time(), f'rate parsed for {n} sites')
    print(get_time(), f'read rates for {len(positions)} positions from .vcf')
    positions = np.array(positions)
    rates = np.array(rates)
    print(get_time(), 'set up position and rate arrays')
    return positions, rates


def main():
    #
    args = get_args()
    L = int(args.L)
    rates = np.zeros(L, dtype=float)
    i = 0

    if args.vcf_fname.endswith('.gz'):
        open_func = gzip.open
    else:
        open_func = open

    with open_func(args.vcf_fname, 'rb') as file:
        for lineb in file:
            line = lineb.decode()
            if line.startswith('#'):
                continue

            split_line = line.strip('\n').split('\t')
            chrom, position, _, ref, alt, __, ___, info = split_line

            if i == 0:
                split_info = info.split(';')
                info_names = [x.split('=')[0] for x in split_info]
                if args.rate_field not in info_names:
                    raise ValueError(
                        f'tag {args.rate_field} not present in info!'
                    )
                rate_idx = info_names.index(args.rate_field)

            # we decrement by 1 to get the 0-indexed position
            idx = int(position) - 1
            if idx < L:
                split_info = info.split(';')
                rates[idx] += float(split_info[rate_idx].split('=')[1])

            i += 1
            if i % args.verbosity == 0:
                print(util.get_time(), f'rate parsed for {i} rows')

    print(util.get_time(), f'read rates for ~{i//3} positions from .vcf')

    rates = rates * coeff
    rates[rates == 0] = np.nan
    # write to file
    np.save(args.out_fname, rates)
    print(util.get_time(), 'wrote mutation rates to file')