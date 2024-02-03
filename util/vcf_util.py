
#

import numpy as np
import gzip
import time


class Columns:

    CHROM_idx = 0
    POS_idx = 1
    ID_idx = 2
    REF_idx = 3
    ALT_idx = 4
    QUAL_idx = 5
    FILTER_idx = 6
    INFO_idx = 7
    FORMAT_idx = 8
    SAMPLE_idx = 9


class Header(Columns):

    def __init__(self, lines, chrom):
        """
        Initialize from a list of bytes lines and a bytes representing the
        chromosome number
        """
        self.chrom = chrom
        if b'##fileformat' in lines[0]:
            self.file_format = lines.pop(0)
        else:
            raise ValueError("No valid ##fileformat line!")
        if b'#CHROM' in lines[-1]:
            self.header_line = lines.pop(-1)
        else:
            raise ValueError("No valid column header line!")
        self.filter_lines = {}
        self.info_lines = {}
        self.format_lines = {}
        self.misc_lines = {}
        self.contig_line = None
        for i in range(len(lines)):
            line = lines.pop(0)
            if b'bcftools' in line:
                pass  # I do not want these retained in files
            elif b'##INFO' in line:
                line_id = self.analyze_line(line)[b'INFO'][b'ID']
                self.info_lines[line_id] = line
            elif b'##FORMAT' in line:
                line_id = self.analyze_line(line)[b'FORMAT'][b'ID']
                self.format_lines[line_id] = line
            elif b'##FILTER' in line:
                line_id = self.analyze_line(line)[b'FILTER'][b'ID']
                self.info_lines[line_id] = line
            elif b'##contig' in line:
                contig_id = self.analyze_line(line)[b'contig'][b'ID']
                if contig_id == self.chrom:
                    self.contig_line = line
            else:
                line_id = self.get_line_name(line)
                self.info_lines[line_id] = line
        if not self.contig_line:
            self.contig_line = self.get_contig_line(self.chrom)

    @classmethod
    def read(cls, path):
        """
        Read a header from a .vcf.gz file
        """
        lines = []
        with gzip.open(path, "r") as file:
            for line in file:
                if b'#' in line:
                    lines.append(line)
                else:
                    chrom = Line(line).chrom
                    break
        return cls(lines, chrom)

    def out(self):
        """
        Return the contents of the instance as a list
        """
        ret = [self.file_format, self.contig_line]
        ret += list(self.info_lines.values())
        ret += list(self.filter_lines.values())
        ret += list(self.format_lines.values())
        ret += list(self.misc_lines.values())
        ret.append(self.header_line)
        return ret

    def simplify(self, format_string, info_string=None, keep_filter=False):

        # parse arguments
        format_list = [x.encode() for x in format_string.split(':')]
        if info_string:
            info_list = [x.decode() for x in info_string.split(';')]
        else:
            info_list = []
        # make a list of lines
        simplified = [self.file_format, self.contig_line]
        if keep_filter:
            simplified += list(self.filter_lines.keys())
        for field in info_list:
            if field in self.info_lines.keys():
                simplified.append(self.info_lines[field])
            else:
                raise ValueError(f"Invalid info field {field}!")
        for field in format_list:
            if field in self.format_lines.keys():
                simplified.append(self.format_lines[field])
            else:
                raise ValueError(f"Invalid format field {field}!")
        simplified.append(self.header_line)
        return Header(simplified, self.chrom)

    @staticmethod
    def get_line_id(line):
        """
        Extract the ID of an INFO or FORMAT header line, eg

        INFO=<ID=XX, ...>\n returns XX
        """
        ret = line.split(b'=')[2]
        ret = ret.split(b',')[0]
        return ret

    @staticmethod
    def analyze_line(line):
        """
        Return a line as a dictionary
        """
        if b'<' in line:
            name, data = line.strip(b'\n').strip(b'>').split(b'<')
            name = name.strip(b'#').strip(b'=')
            data = data.split(b',')
            ret = {name: {}}
            for item in data:
                if b'=' in item:
                    key, value = item.split(b'=', maxsplit=1)
                    ret[name][key] = value
        else:
            name, data = line.strip(b'\n').split(b'=')
            ret = {name: data}
        return ret

    @staticmethod
    def get_line_name(line):
        """
        Return the identifier for a line; the element in this position
        ##LINENAME="...."\n
        """
        return line.split(b'=')[0].strip(b'#')

    @staticmethod
    def get_contig_line(chrom):
        """
        Return a line recording the chromosome number represented in the file
        """
        return b'##contig=<ID=' + chrom + b'>\n'

    @property
    def header_line_as_list(self):
        """
        Return the header line as a list of bytes
        """
        return self.header_line.strip(b'\n').split(b'\t')

    @property
    def sample_ids(self):
        """
        Return the sample_ids as a list of bytes
        """
        return self.header_line_as_list[self.SAMPLE_idx:]

    @property
    def sample_ids_as_str(self):
        """
        Return the sample_ids recorded in the column header line as a list of
        strings
        """
        return [name.decode() for name in self.sample_ids]

    @property
    def chrom_as_int(self):
        """
        Return self.chrom as an integer; self.chrom is read from the first
        non-header line. Use this only if a file contains a single chromosome
        """
        return int(self.chrom.decode())


class Line(Columns):

    def __init__(self, line):
        """
        Initialize from a tab-separated bytes line
        """
        self.fields = line.strip(b'\n').split(b'\t')
        self.n_fields = len(self.fields)

    @classmethod
    def get_first_line(cls, path):
        """
        Scan a vcf.gz file for the first non-header line and return a Line
        instance representing it
        """
        with gzip.open(path, 'r') as in_file:
            for line in in_file:
                if b'#' in line:
                    pass
                else:
                    break
        return cls(line)

    def out(self):
        """
        Return the fields represented in the instance as tab-separated bytes
        """
        return b'\t'.join(self.fields) + b'\n'

    def simplify(self, format_bytes, format_idx, info_idx, keep_id,
                 keep_filter, keep_quality):
        """
        Selectively eliminate information from the specified fields
        """
        if not keep_id:
            self.fields[self.ID_idx] = b'.'
        if not keep_quality:
            self.fields[self.QUAL_idx] = b'.'
        if not keep_filter:
            self.fields[self.FILTER_idx] = b'.'
        if not info_idx:
            self.fields[self.INFO_idx] = b'.'
        else:
            info = self.info_list
            self.fields[self.INFO_idx] = b';'.join([info[i] for i in info_idx])
        self.fields[self.FORMAT_idx] = format_bytes
        for idx in np.arange(9, self.n_fields):
            sample_fields = self.get_split_sample(idx)
            self.fields[idx] = b':'.join(sample_fields[j] for j in format_idx)

    def get_format_idx(self, format_string):
        """
        Return a list indexing the format fields specified in a format string
        of the form GT:GQ:XX
        :type format_string: str
        """
        format_bytes = [x.encode() for x in format_string.split(':')]
        format_idx = [self.format_list.index(x) for x in format_bytes]
        return format_idx

    def get_info_idx(self, info_string):
        """
        Return a list indexing the info fields specified in an info_string of
        the form AF;AA;XX
        :type info_string: str
        """
        info_list = [x.encode() for x in info_string.split(';')]
        info_idx = [self.info_list.index(x) for x in info_list]
        return info_idx

    @property
    def chrom(self):
        """
        Return the chrom specified in the line as bytes
        """
        return self.fields[self.CHROM_idx]

    @property
    def position(self):
        """
        Return the position specified in the line as bytes
        """
        return self.fields[self.POS_idx]

    @property
    def info(self):
        """
        Return the info specified in the line as unaltered bytes
        """
        return self.fields[self.INFO_idx]

    @property
    def info_list(self):
        """
        Return the info specified in the line as a list of bytes
        """
        return self.info.split(b';')

    @property
    def formats(self):
        """
        Return the formats specified in the line as unaltered bytes
        """
        return self.fields[self.FORMAT_idx]

    @property
    def format_list(self):
        """
        Return the formats specified in the line as a list of bytes
        """
        return self.formats.split(b':')

    def get_split_sample(self, idx):
        """
        Split the sample entry indexed in the line by idx into a list
        """
        return self.fields[idx].split(b':')

    def get_format_field_idx(self, field):
        """
        Return an integer that indexes a format field within the list of format
        fields. bytes input
        """
        return self.format_list.index(field)

    @property
    def sample_idx(self):
        """
        Return a range that indexes samples in the line
        """
        return range(self.SAMPLE_idx, self.n_fields)

    def get_genotypes(self):
        """
        Return a list of genotype arrays for each sample in the line
        """
        GT_idx = self.get_format_field_idx(b'GT')
        genotype_bytes = [self.get_split_sample(idx)[GT_idx]
                          for idx in self.sample_idx]
        genotypes = [self.decode_genotype_bytes(genotype)
                     for genotype in genotype_bytes]
        return genotypes

    @staticmethod
    def decode_genotype_bytes(genotype_bytes):
        """
        Decode an unphased genotype represented as bytes of the form b'0/0'
        into an array of the form np.array([[0, 0]], dtype=np.uint8)
        """
        alleles = [allele.decode() for allele in genotype_bytes.split(b'/')]
        return np.array(alleles, dtype=np.uint8)


def simplify(in_path, out_path, format_string, info_string=None, keep_id=False,
             keep_filter=False, keep_quality=False):
    """
    Create an uncompressed copy of a .vcf.gz format file, while selectively
    eliminating information as specified to reduce the file size.

    This results in a sparser file which is easier to read and likely faster
    to manipulate.

    >>> simplify("example.vcf.gz", "out.vcf", "GT:GQ", info_string="DP;AF")

    :param in_path: path to .vcf.gz file
    :param out_path: path to .vcf output file
    :param format_string: colon-separated string of formats to retain
    :param info_string: semicolon-seperated string of info to retain
    """
    # handle arguments
    test_line = Line.get_first_line(in_path)
    format_idx = test_line.get_format_idx(format_string)
    format_bytes = b':'.join([test_line.format_list[i] for i in format_idx])
    if info_string:
        info_idx = test_line.get_info_idx(info_string)
    else:
        info_idx = None
    # set up the header
    header = Header.read(in_path)
    header = header.simplify(format_string, info_string, keep_filter)
    out_file = open(out_path, 'wb')
    for line in header.out():
        # print(line)
        out_file.write(line)
    # loop over every line and write a simplified version to out_path
    i = 0
    with gzip.open(in_path, 'r') as in_file:
        for i, line in enumerate(in_file):
            if b'#' not in line:
                line = Line(line)
                line.simplify(format_bytes, format_idx, info_idx, keep_id,
                              keep_filter, keep_quality)
                out_file.write(line.out())
                i += 0
    out_file.close()
    print(f"{i} lines simplified; written at {out_path}")
    return 0


def read(path):

    header = Header.read(path)
    sample_ids = header.sample_ids_as_str
    n_samples = len(sample_ids)
    positions = []
    genotype_lists = [[] for i in range(n_samples)]
    with gzip.open(path, 'r') as file:
        for i, line in enumerate(file):
            if b'#' not in line:
                line = Line(line)
                positions.append(line.position)
                line_genotypes = line.get_genotypes()
                for idx in range(n_samples):
                    genotype_lists[idx].append(line_genotypes[idx])
    positions = np.array(positions, dtype=np.int64)
    genotypes = {sample_ids[i]: np.array(genotype_lists[i])
                 for i in range(n_samples)}
    return positions, genotypes


def read_positions(path):
    """
    Return a vector of positions from a .vcf.gz file. Positions are 1-indexed
    eg the first position is 1.

    :param path: path to a .vcf.gz file
    :return:
    """
    positions = []
    with gzip.open(path, 'r') as file:
        for line in file:
            if b'#' not in line:
                line = Line(line)
                positions.append(line.position)
    positions = np.array(positions, dtype=np.int64)
    return positions


def read_sample_ids(path):

    header = Header.read(path)
    return header.sample_ids_as_str


def read_chrom(path):

    line = Line.get_first_line(path)
    return int(line.chrom.decode())


def read_first_lines(path, n_lines, fmt='r'):
    """
    Read and return the first k lines of a gzipped file at file_name using
    format 'fmt'. For manual inspection of file headers
    """
    n_lines -= 2
    lines = []
    with gzip.open(path, fmt) as file:
        for i, line in enumerate(file):
            lines.append(line)
            if i > n_lines:
                break
    return lines















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



def _read_sample_ids(path):
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


def _simplify(path, out, *args):
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


def read_sample(path, sample_id):
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
    sample_ids = _read_sample_ids(path)
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
    inverted_ids = {sample_ids[key]: key for key in sample_ids}
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



