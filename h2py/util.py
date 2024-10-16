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

def read_bedfile(fname):
    """
    
    """
    open_func = gzip.open if fname.endswith('.gz') else open
    with open_func(fname, 'rb') as file:
        split_line = file.readline().decode().split('\t')
        skiprows = 0 if split_line[1].isnumeric() else 1
    regions = np.loadtxt(fname, usecols=(1, 2), dtype=int, skiprows=skiprows)
    if regions.ndim == 1:
        regions = regions[np.newaxis]
    return regions


def write_bedfile(fname, chrom_num, regions, header=False):
    """
    
    """
    open_func = gzip.open if fname.endswith('.gz') else open
    with open_func(fname, "wb") as file:
        if header:
            header = b'#chrom\tchromStart\tchromEnd\n'
            file.write(header)
        for start, stop in regions:
            line = f'{chrom_num}\t{start}\t{stop}\n'.encode()
            file.write(line)
    return 


def regions_to_mask(regions, L=None):
    """
    Return a boolean mask array that equals 0 within `regions` and 1 
    elsewhere.
    """
    if L is None:
        L = regions[-1, 1]
    mask = np.ones(L, dtype=bool)
    for (start, end) in regions:
        if start > L:
            break
        if end > L:
            end = L
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


def collapse_regions(elements):
    """
    Collapse any overlapping elements in an array together.
    """
    return mask_to_regions(regions_to_mask(elements))


"""
Recombination maps
"""


def read_recombination_map(fname, L=None, map_col='Map(cM)'):
    """
    Read a recombination map in `hapmap` map format. 
    """
    open_func = gzip.open if fname.endswith('.gz') else open
    with open_func(fname, 'rb') as file:
        header_line = file.readline().decode().split()
    pos_idx = header_line.index('Position(bp)')
    map_idx = header_line.index(map_col)
    mapcoords, mapvals = np.loadtxt(
        fname, skiprows=1, usecols=(pos_idx, map_idx), unpack=True
    )
    assert np.all(np.diff(mapcoords) > 0)
    assert np.all(np.diff(mapvals) >= 0)
    rmap = interpolate.interp1d(mapcoords, mapvals)
    return rmap


def read_bedgraph_recombination_map():
    """
    
    """
    return


"""
Reading .vcf files
"""


def read_genotypes(vcf_fname, bed_fname=None, L=None):
    """
    
    """
    if bed_fname is not None:
        mask = regions_to_mask(read_bedfile(bed_fname))
        if L is not None:
            if len(mask) < L:
                mask = mask[:L]
    elif L is not None:
        mask = 0
    else:
        raise ValueError('you must provide a bedfile or L value')


    open_func = gzip.open if vcf_fname.endswith('.gz') else open
    genotype_arr = []
    positions = []
    with open_func(vcf_fname, "rb") as file:
        for lineb in file:
            line = lineb.decode()
            if line.startswith('#'):
                if line.startswith('##'):
                    continue
                else:
                    sample_ids = line.split()[9:]
                    sample_cols = [line.index(x) for x in sample_ids]
                    continue
            else:
                split_line = line.split()
                position = int(split_line[1]) - 1
                genotypes = [split_line[i].split(':')[0] for i in sample_cols]
                genotype_arr.append()


                positions.append(position)
                

            if i == 0:
                gt_index = fields[format_idx].split(':').index('GT')
            else:
                if i % verbosity == 0:
                    print(get_time(), f'read .vcf row {i}')

            if is_in_mask(position):
                positions.append(position)
                genotypes = []
                for entry in fields[first_sample_idx:]:
                    genotype = entry.split(':')[gt_index]
                    if '/' in genotype:
                        gt = [int(x) for x in genotype.split('/')]
                    elif '|' in genotype:
                        gt = [int(x) for x in genotype.split('|')]
                    else:
                        raise ValueError(r'GT entry has no \ or |')
                    genotypes.append(gt)
                genotype_arr.append(np.array(genotypes))
            i += 1

    positions = np.array(positions, dtype=int)
    genotype_arr = np.stack(genotype_arr, axis=0, dtype=int)
    return sample_ids, positions, genotype_arr


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
    # return a string giving the date and time
    return "[" + datetime.strftime(datetime.now(), "%m-%d-%y %H:%M:%S") + "]"




#####
"""
reading and writing data to file
"""


def read_bedgraph(fname, sep='\t', fields=None):
    # read a .bedgraph
    if fname.endswith('.gz'):
        open_func = gzip.open
    else:
        open_func = open
    with open_func(fname, 'rb') as file:
        header_line = file.readline().decode()

    header = header_line.strip('\n').split(sep)
    data_names = header[3:]
    cols = tuple(range(3, len(header)))

    regions = np.loadtxt(
        fname,
        skiprows=1,
        dtype=int,
        usecols=(1, 2)
    )
    data_cols = np.loadtxt(
        fname,
        skiprows=1,
        usecols=cols,
        dtype=float,
        unpack=True,
    )
    data = dict(zip(data_names, data_cols))
    # add field filter
    return regions, data


def read_fasta_file(fname, map_symbols=True):
    # expects one sequence per file. returns an array of bytes
    if 'gz' in fname:
        open_fxn = gzip.open
    else:
        open_fxn = open
    lines = []
    header = None
    with open_fxn(fname, 'rb') as file:
        for i, line in enumerate(file):
            line = line.rstrip(b'\n')
            if b'>' in line:
                header = line
            else:
                lines.append(line)
    alleles = np.array(list(b''.join(lines).decode()))
    if map_symbols:
        mapping = {'.': 'N', '-': 'N', 'a': 'A', 'g': 'G', 't': 'T', 'c': 'C'}
        for symbol in mapping:
            alleles[alleles == symbol] = mapping[symbol]
    return alleles, header


def get_fa_allele_mask(genotypes):
    # bad name...
    indicator = genotypes != 'N'
    regions = get_mask_from_bool(indicator)
    return regions





def read_vcf_positions(fname):
    # read and return the vector of positions in a .vcf.gz file
    pos_idx = 1
    positions = []
    if ".gz" in fname:
        open_fxn = gzip.open
    else:
        open_fxn = open
    with open_fxn(fname, "rb") as file:
        for line_b in file:
            if line_b.startswith(b'#'):
                continue
            fields = line_b.strip(b'\n').split(b'\t')
            positions.append(int(fields[pos_idx]))
    return np.array(positions)
