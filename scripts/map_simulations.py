"""

"""
import argparse
import demes
import msprime
import numpy as np

from archaic import parsing


def get_args():
    #
    parser = argparse.ArgumentParser()
    parser.add_argument('-r', '--rmap_fname', required=True)
    parser.add_argument('-u', '--umap_fname', required=True)
    parser.add_argument('-b', '--mask_fname', required=True)
    parser.add_argument('--bins', required=True)
    return


