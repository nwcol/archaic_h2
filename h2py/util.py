"""

"""
from datetime import datetime
import gzip
import numpy as np
import numpy.ma as ma
from scipy import interpolate
import re
import warnings


"""
Bedfiles and masks
"""

def read_bedfile(bed_file, min_reg_len=None):
    """
    
    """
    open_func = gzip.open if bed_file.endswith('.gz') else open
    with open_func(bed_file, 'rb') as fin:
        split_line = fin.readline().decode().split()
        skiprows = 0 if split_line[1].isnumeric() else 1
        chrom_num = fin.readline().decode().split()[0]
    regions = np.loadtxt(
        bed_file, usecols=(1, 2), dtype=int, skiprows=skiprows
    )

    if regions.ndim == 1:
        regions = regions[np.newaxis]

    if min_reg_len is not None:
        lens = regions[:, 1] - regions[:, 0]
        regions = regions[lens > min_reg_len]

    return chrom_num, regions


def read_bedfile_positions(bed_file, region=None):
    """
    Return a vector of 0-indexed positions covered in a .bed file.
    """
    chrom_num, regions = read_bedfile(bed_file)
    mask = regions_to_mask(regions)
    positions = np.nonzero(~mask)[0]
    if region is not None:
        within = np.logical_and(positions >= region[0], positions < region[-1])
        positions = positions[within]
    return chrom_num, positions


def write_bedfile(file, chrom_num, regions, header=False):
    """
    
    """
    open_func = gzip.open if file.endswith('.gz') else open
    with open_func(file, "wb") as file:
        if header:
            header = b'#chrom\tchromStart\tchromEnd\n'
            file.write(header)
        for start, stop in regions:
            line = f'{chrom_num}\t{start}\t{stop}\n'.encode()
            file.write(line)
    return 


def regions_to_mask(regions, length=None):
    """
    Return a boolean mask array that equals 0 within `regions` and 1 
    elsewhere.
    """
    if length is None:
        length = regions[-1, 1]
    mask = np.ones(length, dtype=bool)
    for (start, end) in regions:
        if start >= length:
            continue
        elif end > length:
            end = length
        mask[start:end] = 0
    return mask


def mask_to_regions(mask):
    """
    Return an array representing the regions that are not masked in a boolean
    array (0s).
    """
    jumps = np.diff(np.concatenate(([1], mask, [1])))
    starts = np.where(jumps == -1)[0]
    ends = np.where(jumps == 1)[0]
    regions = np.stack([starts, ends], axis=1)
    return regions


def intersect_bed_regions(*bed_regions):
    """

    """
    length = max([reg[-1, 1] for reg in bed_regions])
    masks = [regions_to_mask(reg, length=length) for reg in bed_regions]
    sums = np.sum(masks, axis=1)
    overlap_mask = sums < 0
    regions = mask_to_regions(overlap_mask)
    return regions


def collapse_regions(elements):
    """
    Collapse any overlapping elements in an array together.
    """
    return mask_to_regions(regions_to_mask(elements))


def read_bedgraph(file, sep=','):
    """
    From a bedgraph-format file, read and return chromosome number(s), an 
    array of genomic regions and a dictionary of data columns. 

    If the file has one unique chromosome number, returns it as a string of
    the form `chr00`; if there are several, returns an array of string
    chromosome numbers of this form for each row.
    Possible file extensions include but are not limited to .bedgraph, .csv,
    and .tsv, with column seperator determined by the `sep` argument.
    """
    open_func = gzip.open if file.endswith('.gz') else open
    with open_func(file, 'rb') as file:
        header_line = file.readline().decode().strip().split(sep)
    # check for proper header format
    assert header_line[0] in ['chrom', '#chrom']
    assert header_line[1] in ['chromStart', 'start']
    assert header_line[2] in ['chromEnd', 'end']
    fields = header_line[3:]
    # handle the return of the chromosome number(s)
    chrom_nums = np.loadtxt(
        file, usecols=0, dtype=str, skiprows=1, delimiter=sep
    )
    if len(set(chrom_nums)) == 1:
        ret_chrom = chrom_nums[0]
    else:
        # return the whole vector if there are >1 unique chromosome
        ret_chrom = chrom_nums
    windows = np.loadtxt(
        file, usecols=(1, 2), dtype=int, skiprows=1, delimiter=sep
    )
    cols_to_load = tuple(range(3, len(header_line)))
    arr = np.loadtxt(
        file,
        usecols=cols_to_load,
        dtype=float,
        skiprows=1,
        unpack=True,
        delimiter=sep
    )
    dataT = [arr] if arr.ndim == 1 else [col for col in arr]
    data = dict(zip(fields, dataT))
    return ret_chrom, windows, data


def write_bedgraph(file, chrom_num, regions, data, sep=','):
    """
    Write a .bedgraph-format file from an array of regions/windows and a 
    dictionary of data columns.
    """
    for field in data:
        if len(data[field]) != len(regions):
            raise ValueError(f'data field {data} mismatches region length!')
    open_func = gzip.open if file.endswith('.gz') else open
    fields = list(data.keys())
    header = sep.join(['#chrom', 'chromStart', 'chromEnd'] + fields) + '\n'
    with open_func(file, 'wb') as file:
        file.write(header.encode())
        for i, (start, end) in enumerate(regions):
            ldata = [str(data[field][i]) for field in fields]
            line = sep.join([chrom_num, str(start), str(end)] + ldata) + '\n'
            file.write(line.encode())
    return


"""
Recombination maps
"""


def read_recombination_map(rec_map_file, map_col='Map(cM)'):
    """
    Read a recombination map in `hapmap` map format. 
    """
    open_func = gzip.open if rec_map_file.endswith('.gz') else open
    with open_func(rec_map_file, 'rb') as fin:
        header_line = fin.readline().decode().split()
    pos_idx = header_line.index('Position(bp)')
    map_idx = header_line.index(map_col)
    mapcoords, mapvals = np.loadtxt(
        rec_map_file, skiprows=1, usecols=(pos_idx, map_idx), unpack=True
    )
    assert np.all(np.diff(mapcoords) > 0)
    assert np.all(np.diff(mapvals) >= 0)
    rmap = interpolate.interp1d(mapcoords, mapvals, fill_value=0)
    return rmap


def read_recombination_map(rec_map_file, positions, map_col='Map(cM)'):
    """
    Read a recombination map in `hapmap` map format and interpolate map values
    for a vector of positions.
    """
    open_func = gzip.open if rec_map_file.endswith('.gz') else open
    with open_func(rec_map_file, 'rb') as fin:
        header_line = fin.readline().decode().split()
    pos_idx = header_line.index('Position(bp)')
    map_idx = header_line.index(map_col)
    map_coords, map_vals = np.loadtxt(
        rec_map_file, skiprows=1, usecols=(pos_idx, map_idx), unpack=True
    )
    assert np.all(np.diff(map_coords) > 0)
    assert np.all(np.diff(map_vals) >= 0)
    pos_map = np.interp(positions, map_coords, map_vals)
    return pos_map


def read_bedgraph_recombination_map():
    """
    
    """
    return


def read_mutation_map(mut_map_file, positions):
    """
    """
    if (
        mut_map_file.endswith('.bedgraph') 
        or mut_map_file.endswith('.bedgraph.gz')
    ):
        _, regions, data = read_bedgraph()
        # interpolate.
        idxs = np.searchsorted(regions[:, 1], positions)
        reg_mut_map = data['u']
        mut_map = reg_mut_map[idxs]
        
    elif mut_map_file.endswith('.npy'):
        tot_mut_map = np.load(mut_map_file)
        mut_map = tot_mut_map[positions]
        assert not np.any(np.isnan(mut_map))

    else:
        raise ValueError('unrecognized mutation map format')

    return mut_map


"""
Reading .vcf files
"""


def read_genotypes(
    vcf_file, 
    bed_file=None, 
    min_reg_len=None,
    region=None,
    ancestral_seq=None, 
    read_multiallelic=False
):
    """
    Read a genotype matrix from a .vcf file. Matrix has shape 
    (nsamples, nsites, 2). Ignores sites that are not biallelic. 
    """
    if bed_file is not None:
        _, regions = read_bedfile(bed_file, min_reg_len=min_reg_len)
        mask = regions_to_mask(regions)
        len_mask = len(mask)
    open_func = gzip.open if vcf_file.endswith('.gz') else open

    outside_mask = 0
    outside_reg = 0
    multiallelic = 0

    chrom_nums = []
    positions = []
    genotypes = []

    with open_func(vcf_file, "rb") as fin:
        for lineb in fin:
            line = lineb.decode()
            if line.startswith('#'):
                if line.startswith('#CHROM'):
                    sample_ids = line.split()[9:]
                    num_samples = len(sample_ids)
            else:
                split_line = line.split()
                chrom, pos, _, ref, alt = split_line[:5]
                samples = split_line[9:]
                position = int(pos) - 1

                if region is not None:
                    if position < region[0] or position >= region[-1]:
                        outside_reg += 1
                        continue

                if bed_file is not None:
                    if position >= len_mask or mask[position] == 1:
                        outside_mask += 1
                        continue

                if not read_multiallelic:
                    if len(alt.split(',')) > 1:
                        multiallelic += 1
                        continue

                line_genotypes = np.array(
                    [re.split('/|\|', s.split(':')[0]) for s in samples],
                    dtype=np.int64
                )
                chrom_nums.append(chrom)
                positions.append(position)
                genotypes.append(line_genotypes)

    positions = np.array(positions, dtype=np.int64)
    genotype_matrix = np.stack(genotypes, axis=1, dtype=np.int64)
    unique_chrom_nums = list(set(chrom_nums))
    if len(unique_chrom_nums) > 1:
        warnings.warn('more than one unique chromosome in .vcf')
    chrom_num = unique_chrom_nums[0]

    return chrom_num, sample_ids, positions, genotype_matrix


"""
Math
"""


def n_choose_2(n):
    """
    
    """
    return n * (n - 1) // 2


"""
Recombination map arithmetic
"""


def map_function(r):
    """
    Haldane's map function; transforms distance in r to cM.
    """
    return -50 * np.log(1 - 2 * r)


def inverse_map_function(d):
    """
    The inverse of Haldane's map function. Transforms distance in cM to r.
    """
    return (1 - np.exp(-d / 50)) / 2


"""
printouts
"""


def get_time():
    """
    Return a string giving the time and date with yy-mm-dd format
    """
    return "[" + datetime.strftime(datetime.now(), "%y-%m-%d %H:%M:%S") + "]"
