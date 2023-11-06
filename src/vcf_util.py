
# Utilities for parsing information from .vcf.gz format files

import numpy as np

import gzip


def read_lines(file_name, k, form='r'):
    """
    Read and return the first k lines of a gzipped file at file_name using
    format 'form'.

    :param file_name: file to read
    :param k: number of lines to read
    :param form: return bytes if 'r' and strings if 'rt'
    :return:
    """
    k -= 2
    lines = []
    with gzip.open(file_name, form) as file:
        for i, line in enumerate(file):
            lines.append(line)
            if i > k:
                break
    return lines


def parse_header(input_filename):
    headers = []
    with gzip.open(input_filename, "r") as file:
        for line in file:
            if b'#' in line:
                headers.append(line)
            else:
                break
    return headers


def trim_header(input_filename, fields):
    headers = parse_header(input_filename)
    file_format = headers.pop(0)
    column_titles = headers.pop(-1)
    field_names = [b'##FORMAT=<ID=' + field for field in fields]
    field_lines = []
    for line in headers:
        for name in field_names:
            if name in line:
                field_lines.append(line)
    header = [file_format] + field_lines + [column_titles]
    return header


def parse_format(input_filename, formats):
    format_col = 8
    with gzip.open(input_filename, "r") as file:
        for line in file:
            if b'#' not in line:
                format_fields = line.split()[format_col].split(b':')
                break
    format_index = []
    for field in formats:
        format_index.append(format_fields.index(field))
    sort = np.argsort(format_index)
    sorted_formats = [formats[i] for i in sort]
    format_index = [format_index[i] for i in sort]
    return format_index, sorted_formats


def parse_samples(input_filename):
    """
    Return a list of sample names in a .vcf.gz file in bytes format. With
    only one sample, a singleton list is returned.

    :param input_filename:
    :return:
    """
    sample_col_0 = 9
    with gzip.open(input_filename, "r") as file:
        for line in file:
            if b'#CHROM' in line:
                columns = line.split()
                break
    samples = columns[sample_col_0:]
    return samples


def get_sample_index(input_filename):
    n_samples = len(parse_samples(input_filename))
    index = list(range(9, 9 + n_samples))
    return index


def get_format_bytes(sorted_formats):
    return b':'.join(sorted_formats)


def simplify_line(line, format_bytes, format_index, sample_index):
    format_col = 8
    elements = line.split()
    elements[format_col] = format_bytes
    for i in sample_index:
        data = elements[i].split(b':')
        elements[i] = b':'.join([data[j] for j in format_index])
    line = b'\t'.join(elements) + b'\n'
    return line


def count_positions(file_name):
    """
    Count the number of positions recorded in a .vcf.gz file

    :param file_name:
    :return:
    """
    with gzip.open(file_name, 'r') as file:
        count = np.sum([1 for line in file if b'#' not in line])
    return count


def count_lines(file_name):
    """
    Count the number of lines in a .vcf.gz file

    :param file_name:
    :return:
    """
    with gzip.open(file_name, 'r') as file:
        count = np.sum([1 for line in file])
    return count


def read_genotypes(file_name, sample):
    """
    Return a vector of alternate allele counts for a sample in a .vcf.gz file

    :param file_name:
    :param sample:
    :return:
    """
    sample = sample.encode()
    sample_col_0 = 9
    samples = parse_samples(file_name)
    sample_index = samples.index(sample) + sample_col_0
    n_positions = count_positions(file_name)
    alts = np.zeros(n_positions, dtype=np.uint8)
    i = 0
    with gzip.open(file_name, 'r') as file:
        for line in file:
            if b'#' not in line:
                alts[i] = parse_genotype(line, sample_index)
                i += 1
    return alts


def read_genotype(line, sample_index, geno_index=0):
    """
    Read the genotype in a line and return it as bytes

    :param line:
    :param geno_index: index of the genotype in the sample column
    :return:
    """
    sample = line.split()[sample_index]
    genotype = sample.split(b':')[geno_index]
    return genotype


def parse_genotype(line, sample_index, geno_index=0):
    """
    Parse the genotype in a line and return 0 for reference homozygote,
    1 for heterozygote, 2 for alternate homozygote

    :param line:
    :param geno_index: index of the genotype in the sample column
    :return:
    """
    genotype = read_genotype(line, sample_index, geno_index=0)
    code = {b'0/0': 0, b'0/1': 1, '1/0': 1, b'1/1': 2, b'1/2': 2}[genotype]
    return code


def scan_genotypes(file_name, sample):
    """
    Count the numbers of genotypes in a .vcf.gz file

    :param file_name:
    :param sample:
    :return:
    """
    counts = {b'0/0': 0, b'0/1': 0, b'1/0': 0, b'1/1': 0, b'1/2': 0}
    sample_col_0 = 9
    samples = parse_samples(file_name)
    sample_index = samples.index(sample) + sample_col_0
    i = 0
    with gzip.open(file_name, 'r') as file:
        for line in file:
            if b'#' not in line:
                geno = parse_genotype(line, sample_index)
                counts[geno] += 1
                i += 1
    return counts


def parse_chrom(file_name):
    """
    Return the chromosome number as a string

    :param file_name:
    :return:
    """
    chrom_col = 0
    with gzip.open(file_name, "r") as file:
        for line in file:
            if b'#' not in line:
                chrom = line.split()[chrom_col].decode()
                break
    return chrom


def parse_position(line):
    """
    Return an integer position value for a .vcf file line

    :param line:
    :return:
    """
    pos_index = 1
    position = int(line.split()[pos_index])
    return position






