import numpy as np

import archaic.bed_util as bed_util


def main():
    print("looks like im working!!!!")
    print(bed_util.Bed(np.array([[0, 100]]), "hello").regions)
