
# Utilities for parsing information from .vcf.gz format files

import numpy as np

import gzip


# names the 8 mandatory .vcf format columns and two additional useful ones
vcf_cols = {"#CHROM": 0,
            "POS": 1,
            "ID": 2,
            "REF": 3,
            "ALT": 4,
            "QUAL": 5,
            "FILTER": 6,
            "INFO": 7,
            "FORMAT": 8,
            "sample_0": 9}


def read_first_lines(path, k, fmt='r'):
    """
    Read and return the first k lines of a gzipped file at file_name using
    format 'fmt'. For manual inspection of file headers

    :param path: path to file
    :param k: number of lines to read
    :param fmt: return bytes if 'r' and strings if 'rt'
    :return:
    """
    k -= 2
    lines = []
    with gzip.open(path, fmt) as file:
        for i, line in enumerate(file):
            lines.append(line)
            if i > k:
                break
    return lines


def read_header(path):
    """
    Read and return header lines (lines containing b'#') as a list of bytes
    objects

    :param path:
    :return:
    """
    headers = []
    with gzip.open(path, "r") as file:
        for line in file:
            if b'#' in line:
                headers.append(line)
            else:
                break
    return headers


def trim_header(path, fields):
    """
    Clear a header of all lines which do not specify a field specified in the
    fields argument

    :param path:
    :param fields:
    :return:
    """
    headers = read_header(path)
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


def read_format(path, formats):
    """


    :param path:
    :param formats:
    :return:
    """
    format_col = 8
    with gzip.open(path, "r") as file:
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


def read_samples(path):
    """
    Return a list of sample names in a .vcf.gz file in bytes format. With
    only one sample, a singleton list is returned.

    :param path:
    :return:
    """
    with gzip.open(path, "r") as file:
        for line in file:
            if b'#CHROM' in line:
                columns = line.split()
                break
    samples = columns[vcf_cols["sample_0"]:]
    return samples


def get_sample_index(path):
    n_samples = len(read_samples(path))
    index = list(range(vcf_cols["sample_0"], vcf_cols["sample_0"] + n_samples))
    return index


def get_format_bytes(sorted_formats):
    return b':'.join(sorted_formats)


def simplify_line(line, format_bytes, format_index, sample_index):
    """
    Simplify one line

    :param line:
    :param format_bytes:
    :param format_index:
    :param sample_index:
    :return:
    """
    elements = line.split()
    elements[vcf_cols["REF"]] = b'.'
    elements[vcf_cols["INFO"]] = b'.'
    elements[vcf_cols["FORMAT"]] = format_bytes
    for i in sample_index:
        data = elements[i].split(b':')
        elements[i] = b':'.join([data[j] for j in format_index])
    line = b'\t'.join(elements) + b'\n'
    return line


def count_positions(path):
    """
    Count the number of positions recorded in a .vcf.gz file

    :param path:
    :return:
    """
    with gzip.open(path, 'r') as file:
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


def read_genotypes(path, sample, n_positions=None):
    """
    Return a vector of alternate allele counts for a sample in a .vcf.gz file

    Maps any heterozygous genotype to 1 and any homozygous genotype to 0

    :param path: path to a .vcf.gz file
    :param sample: the sample in the .vcf.gz to be read
    :type sample: string
    :return:
    """
    sample = sample.encode()
    samples = read_samples(path)
    column = samples.index(sample) + vcf_cols["sample_0"]
    if not n_positions:
        n_positions = count_positions(path)
    alts = np.zeros(n_positions, dtype=np.uint8)
    i = 0
    with gzip.open(path, 'r') as file:
        for line in file:
            if b'#' not in line:
                genotype = parse_genotype(line, column)
                alts[i] = eval_genotype(genotype)
                i += 1
    return alts


def parse_position(line):
    """
    Return an integer position value for a .vcf file line

    :param line:
    :return:
    """
    position = int(line.split()[vcf_cols["POS"]])
    return position


def parse_genotype(line, column):
    """
    Parse the genotype from a line and return it as a bytes object. Assumes
    that the genotype is the first field appearing in the sample column.

    :param line:
    :param column: column to parse, 0-indexed
    :return: genotype as bytes
    """
    sample = line.split()[column]
    genotype = sample.split(b':')[0]
    return genotype


def eval_genotype(genotype):
    """
    Evaluate a genotype, returning a 0 if homozygous for the reference, 1 if
    heterozygous in any combination, 2 if homozygous for an alternate allele

    :param genotype: bytes object of the form b'0/0' or b'0|0'
    :return:
    """
    if b'/' in genotype:
        allele_0, allele_1 = genotype.split(b'/')
    elif b'|' in genotype:
        allele_0, allele_1 = genotype.split(b'|')
    else:
        raise ValueError(f"invalid genotype format: {genotype}")
    code = None
    if allele_0 == allele_1:
        if allele_0 == b'0':
            code = 0
        elif allele_1 != b'0':
            code = 2
    elif allele_0 != allele_1:
        code = 1
    return code


def count_genotypes(path, sample):
    """
    Count the numbers of each genotype state present in a .vcf.gz file

    :param path:
    :param sample:
    :return:
    """
    sample = sample.encode()
    samples = read_samples(path)
    column = samples.index(sample) + vcf_cols["sample_0"]
    counts = dict()
    with gzip.open(path, 'r') as file:
        for line in file:
            if b'#' not in line:
                genotype = parse_genotype(line, column)
                if genotype not in counts:
                    counts[genotype] = 1
                else:
                    counts[genotype] += 1
    return counts


def read_chrom(file_name):
    """
    Return the first chromosome number in a .vcf file as a string

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


def count_hets(path, sample):
    """
    Tally and return the number of heterozygous sites and total number of sites
     in a .vcf.gz file

    :param path: path to .vcf.gz file
    :param sample: sample for which to tally heterozygous sites
    :return: number of heterozygous sites
    """
    sample = sample.encode()
    samples = read_samples(path)
    col = samples.index(sample) + vcf_cols["sample_0"]
    h = 0
    i = 0
    with gzip.open(path, 'r') as file:
        for line in file:
            if b'#' not in line:
                genotype = parse_genotype(line, col)
                if eval_genotype(genotype) == 1:
                    h += 1
                i += 1
    return h, i
