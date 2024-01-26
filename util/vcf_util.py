
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


def read_chrom(file_name):
    """
    Return the first chromosome number in a .vcf file as bytes

    :param file_name:
    :return:
    """
    chrom_col = 0
    with gzip.open(file_name, "r") as file:
        for line in file:
            if b'#' not in line:
                chrom = line.split()[chrom_col]
                break
    return chrom


def read_header(path):
    """
    Read and return header lines (lines containing b'#') as a list of bytes
    objects

    :param path:
    :return:
    """
    header = []
    with gzip.open(path, "r") as file:
        for line in file:
            if b'#' in line:
                header.append(line)
            else:
                break
    return header


def read_format_fields(path):
    """
    Read the format fields in the first and return them as a dictionary

    :param path:
    :return:
    """
    formats = []
    with gzip.open(path, "r") as file:
        for line in file:
            if b'##' in line:
                if b"FORMAT" in line:
                    formats.append(line)
            else:
                if b'#' in line:
                    pass
                else:
                    format_col = line.split()[vcf_cols["FORMAT"]]
                    break
    format_dict = dict()
    for i, field in enumerate(format_col.split(b':')):
        format_dict[field] = i
    return format_dict


def read_sample_ids(path):
    """
    Return a dictionary of sample names (bytes formats) defining the
    column indices where they can be found.

    :param path:
    :return:
    """
    with gzip.open(path, "r") as file:
        for line in file:
            if b'#CHROM' in line:
                columns = line.split()
                break
    sample_id_dict = dict()
    for i, column in enumerate(columns):
        if i >= vcf_cols["sample_0"]:
            sample_id_dict[column] = i
    return sample_id_dict


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
        elif allele_0 == b'.':
            code = -1
        elif allele_1 != b'0':
            code = 2
    elif allele_0 != allele_1:
        code = 1
    return code


def simplify_header(header, format_fields):
    """
    Simplify a header

    :param header: list of bytes; the original file header
    :param format_fields: list of bytes; the formats to retain in the header
    :return:
    """
    simplified_header = [header.pop(0)]
    column_titles = header.pop(-1)
    for line in header:
        if b'##FORMAT' in line:
            for field in format_fields:
                if b'##FORMAT=<ID=' + field in line:
                    simplified_header.append(line)
        elif b'##FILTER' in line:
            simplified_header.append(line)
        else:
            pass
    simplified_header.append(column_titles)
    return simplified_header


def simplify_line(line, format_fields, format_index, sample_index):
    """
    Simplify one line

    :param line:
    :param format_fields:
    :type format_fields: bytes
    :param format_index:
    :param sample_index:
    :return:
    """
    elements = line.split()
    elements[vcf_cols["ID"]] = b'.'
    elements[vcf_cols["INFO"]] = b'.'
    elements[vcf_cols["FORMAT"]] = format_fields
    for i in sample_index:
        data = elements[i].split(b':')
        elements[i] = b':'.join([data[j] for j in format_index])
    line = b'\t'.join(elements) + b'\n'
    return line


# Important functions


def simplify(path, out, *args):
    """
    Write a .vcf copy of a .vcf.gz file, removing everything from the ID and
    INFO columns and truncating FORMAT to the fields specified in *args

    I output files as .vcf because I do not know how to produce a gzipped
    file from within python.

    :param path: path to .vcf.gz file
    :param out: name of the .vcf output file
    :param args:
    :return:
    """
    format_fields = [arg.encode() for arg in args]
    format_dict = read_format_fields(path)
    for field in format_fields:
        if field not in format_dict:
            raise ValueError(f"{field} is not a valid format field")
    format_index = [format_dict[key] for key in format_dict
                    if key in format_fields]
    format_col = b':'.join(format_fields)
    sample_index = list(read_sample_ids(path).values())
    header = read_header(path)
    simplified_header = simplify_header(header, format_fields)
    if not out:
        out = path.strip(".gz")
    out_file = open(out, 'wb')
    for line in simplified_header:
        out_file.write(line)
    with gzip.open(path, 'r') as file:
        for line in file:
            if b"#" not in line:
                out_file.write(
                    simplify_line(line, format_col, format_index, sample_index)
                )
    out_file.close()
    return 0


def read_sample(path, sample_id, n_positions=None):
    """
    Return a vector of alternate allele counts for a sample in a .vcf.gz file

    Maps any heterozygous genotype to 1 and any homozygous genotype to 0

    :param path: path to a .vcf.gz file
    :param sample: the sample in the .vcf.gz to be read
    :type sample: string
    :param n_positions:
    :return:
    """
    sample_id = sample_id.encode()
    samples = read_sample_ids(path)
    column = samples[sample_id]
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


def read_samples(path):
    """
    Read the genotypes of every sample in a .vcf and return them bundled in
    dictionary along with a vector of positions.

    :param path:
    :return:
    """
    sample_ids = read_sample_ids(path)
    n_samples = len(sample_ids)
    positions = read_positions(path)
    n_positions = len(positions)
    genotype_arr = np.zeros((n_positions, n_samples), dtype=np.uint8)
    i = 0
    with gzip.open(path, 'r') as file:
        for line in file:
            if b'#' not in line:
                genotypes = parse_genotypes(line)
                genotype_arr[i] = eval_genotypes(genotypes)
                i += 1
    inverted_ids = {sample_ids[key]:key for key in sample_ids}
    sample_dict = {}
    for i in np.arange(n_samples):
        column = 9 + i
        sample = inverted_ids[column]
        sample_dict[sample] = genotype_arr[:, i]
    return sample_dict, positions


def parse_genotypes(line):
    """


    :param line:
    :return:
    """
    genotypes = line.split()[9:]
    return genotypes


def eval_genotypes(genotypes):
    """


    :param genotypes:
    :return:
    """
    codes = np.zeros(len(genotypes), dtype=np.uint8)
    for i, genotype in enumerate(genotypes):
        codes[i] = eval_genotype(genotype)
    return codes


def read_positions(path):
    """
    Return a vector of positions from a .vcf.gz file. Positions are 1-indexed
    eg the first position is 1.

    :param path: path to a .vcf.gz file
    :param n_positions:
    :return:
    """
    positions = []
    with gzip.open(path, 'r') as file:
        for line in file:
            if b'#' not in line:
                positions.append(parse_position(line))
    positions = np.array(positions, dtype=np.int64)
    return positions


def count_genotypes(path, sample):
    """
    Count the numbers of each genotype state present in a .vcf.gz file

    :param path:
    :param sample:
    :return:
    """
    sample = sample.encode()
    samples = read_sample_ids(path)
    column = samples[sample]
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


def count_hets(path, sample):
    """
    Tally and return the number of heterozygous sites and total number of sites
     in a .vcf.gz file

    :param path: path to .vcf.gz file
    :param sample: sample for which to tally heterozygous sites
    :return: number of heterozygous sites
    """
    sample = sample.encode()
    samples = read_sample_ids(path)
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
