
# for loading and handling FASTA files (.fa, .fasta etc)

import gzip
import numpy as np


class FastaArr:

    def __init__(self, seq, chrom):
        # seq is an array of unsigned 8bit ints
        # it is 0-indexed eg the first site has index 0
        self.seq = seq
        self.chrom = chrom
        # properties
        self.length = len(seq)

    @classmethod
    def read_gz(cls, in_file_name):
        # read zipped file
        lines = []
        with gzip.open(in_file_name, 'r') as file:
            for i, line in enumerate(file):
                lines.append(line.rstrip(b'\n'))
        header = lines.pop(0)
        seq_list = [nuc for line in lines for nuc in line]
        seq_arr = np.array(seq_list, dtype=np.uint8)
        # seq = b"".join(lines)
        chrom = header.lstrip(b">chr").decode()
        return cls(seq_arr, chrom)

    def get_motif_start_idx(self, motif):

        motif = np.array(list(motif.encode()), dtype=np.uint8)
        n = self.length
        len_motif = len(motif)
        col_idx = np.arange(len_motif)
        frames = self.seq[np.arange(n - len_motif + 1)[:, None] + col_idx]
        mask = np.all(frames == motif, axis=1)
        idx = np.nonzero(mask)[0]
        return idx

    def get_motif_positions(self, motif):
        # 0 indexed
        start_idx = self.get_motif_start_idx(motif)
        len_motif = len(motif)
        spots = np.array([start_idx, start_idx + len_motif - 1]).T
        mask = np.zeros(self.length, dtype=np.uint8)
        mask[spots] = 1
        idx = np.nonzero(mask)[0]
        return idx

    def find_motif_unvec(self, motif):

        motif = np.array(list(motif.encode()), dtype=np.uint8)
        motif_length = len(motif)
        n = self.length
        raw_idx = [i for i in np.arange(n - motif_length)
                   if np.all(self.seq[i:i + motif_length] == motif)]
        return raw_idx


    def find0_motif(self, motif):
        # motif is string
        motif = np.array(list(motif.encode()), dtype=np.uint8)
        motif_length = len(motif)
        n = self.length
        frames = np.zeros((n - motif_length + 1, motif_length), dtype=np.uint8)
        for i in range(motif_length):
            frames[:, i] = self.seq[i:n - motif_length + i + 1]
        raw_idx = np.nonzero(frames == motif)[0]
        row_idx = np.array(list(set(raw_idx)), dtype=np.int64)
        row_idx = np.sort(row_idx)
        return row_idx


def load(file_name):

    lines = []
    with gzip.open(file_name, 'r') as file:
        for i, line in enumerate(file):
            lines.append(line.rstrip(b'\n'))
    header = lines.pop(0)
    seq_list = [nuc for line in lines for nuc in line]
    seq_arr = np.array(seq_list, dtype=np.uint8)
    return seq_arr, header


def test(file_name, maxx):

    ret = []
    with gzip.open(file_name, 'r') as file:
        for i, line in enumerate(file):
            ret.append(line)
            if i > maxx:
                break
    return ret
