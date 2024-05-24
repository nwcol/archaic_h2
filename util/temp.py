
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
from util import one_locus
from util import two_locus


plt.rcParams['figure.dpi'] = 100
matplotlib.use('Qt5Agg')


"""
H2 vs H^2
"""


def n_choose_2(n):

    return int(n * (n - 1) * 0.5)


def get_H_squared(archive, sample_idx):

    H = archive["H_counts"][sample_idx].sum() / archive["site_counts"].sum()
    return H ** 2


def get_adjusted_H_squared(archive, sample_idx):
    # H2 = h(h - 1) / s(s - 1)
    H_count = archive["H_counts"][sample_idx].sum()
    site_count = archive["site_counts"].sum()
    return (H_count * (H_count - 1)) / (site_count * (site_count - 1))


def get_H2(archive, sample_idx):

    site_pair_counts = archive["site_pair_counts"].sum(0)
    H2 = archive["H2_counts"][sample_idx].sum(0) / site_pair_counts
    return H2


def get_tot_H2(archive, sample_idx):

    site_pair_counts = archive["site_pair_counts"].sum()
    tot_H2 = archive["H2_counts"][sample_idx].sum() / site_pair_counts
    return tot_H2

