
# Functions for computing statistics from vectors of alternate allele counts
# Currently limited to diversity, divergence

import gzip

import matplotlib.pyplot as plt

import matplotlib

import numpy as np


def compute_diversity(alt_x, n_x):
    """
    Compute an estimator for diversity from a vector of alternate allele counts

    :param alt_x:
    :param n_x:
    :return:
    """
    length = len(alt_x)
    ref_x = n_x - alt_x
    tot = alt_x * ref_x
    coeff = 2 / (length * n_x * (n_x - 1))
    diversity = coeff * np.sum(tot)
    return diversity


def compute_divergence(alt_x, alt_y, n_x, n_y):
    """
    Compute an estimator for divergence from two vectors of alternate allele
    counts.

    :param alt_x:
    :param alt_y:
    :param n_x:
    :param n_y:
    :return:
    """
    length = len(alt_x)
    if len(alt_y) != length:
        raise ValueError("Alt vector lengths do not match")
    ref_x = n_x - alt_x
    ref_y = n_y - alt_y
    tot = (ref_x * alt_y) + (alt_x * ref_y)
    coeff = 1 / (length * n_x * n_y)
    divergence = coeff * np.sum(tot)
    return divergence


def compute_F_2(pi_1, pi_2, pi_12):
    """
    Compute an estimator for the F2 statistic

    :return:
    """
    F_2 = pi_12 - (pi_1 + pi_2) / 2
    return F_2


def compute_F_3(pi_x, pi_1, pi_2, pi_x1, pi_x2, pi_12):
    """
    Compute an estimator for the F_3 statistic

    :param pi_x:
    :param pi_1:
    :param pi_2:
    :param pi_x1:
    :param pi_x2:
    :param pi_12:
    :return:
    """
    F_2_x1 = compute_F_2(pi_x1, pi_x, pi_1)
    F_2_x2 = compute_F_2(pi_x2, pi_x, pi_2)
    F_2_12 = compute_F_2(pi_12, pi_1, pi_2)
    F_3 = 0.5 * F_2_x1 * F_2_x2 * F_2_12
    return F_3


def compute_F_st():
    pass












































