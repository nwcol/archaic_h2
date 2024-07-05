"""
mostly exploratory
"""

import demes
import matplotlib.pyplot as plt
import matplotlib
import msprime
import numpy as np
import scipy
from archaic import plotting


"""
Coalescent rates
"""


def get_coalescent_rate(graph, t, sample_name, n=2):

    demography = msprime.Demography.from_demes(graph)
    debugger = demography.debug()
    rates, probs = debugger.coalescence_rate_trajectory(t, {sample_name: n})
    return rates


"""
Recombination 
"""


def E_recombinations(r, t, t_gen=30):
    # expected number of recombinations
    E_rec = (t / t_gen) * r
    return E_rec


def E_time_to_recombination(r, t_gen=30):
    # expected time to a single recombination
    E_t = t_gen * 1 / r
    return E_t


def plot_recomb(times, t_gen=30):

    r = np.logspace(-6, -2, 100)
    fig, ax = plt.subplots(figsize=(6, 5), layout="constrained")
    ax.grid(alpha=0.2)
    ax.set_ylabel("E[recombinations]")
    ax.set_xlabel("r")
    ax.set_yscale("log")
    ax.set_xscale("log")
    colors = plotting.get_gnu_cmap(len(times))
    for i, t in enumerate(times):
        gens = t / t_gen
        exp_recombs = gens * r
        plt.plot(r, exp_recombs, color=colors[i], label=t)
    ax.legend()


"""
phase-type distributions
"""


def phase_solver(alpha, T, i=20):

    p = len(alpha)
    t = - T @ np.ones(p)
    Ti = np.linalg.matrix_power(T, i)
    i_fac = scipy.special.gamma(i + 1)
    return lambda s: s ** i / i_fac * alpha @ Ti @ t


"""
Two-locus process - Markov chain. Exploratory.
"""


def get_P(r, Ne):

    R = 2 * r * Ne
    P = np.array([
        [0, (2 * R) / (1 + 2 * R), 0, 0, 0, 0, 0, 0, 1 / (1 + 2 * R)],
        [1 / (3 + R), 0, R / (3 + R), 1 / (3 + R), 1 / (3 + R), 0, 0, 0, 0],
        [0, 2 / 3, 0, 0, 0, 1 / 6, 1 / 6, 0, 0],
        [0, 0, 0, 0, 0, R / (1 + R), 0, 0, 1 / (1 + R)],
        [0, 0, 0, 0, 0, 0, R / (1 + R), 0, 1 / (1 + R)],
        [0, 0, 0, 2 / 3, 0, 0, 0, 1/ 3, 0],
        [0, 0, 0, 0, 2 / 3, 0, 0, 1 / 3, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 1],
        [0, 0, 0, 0, 0, 0, 0, 0, 1]
    ])
    return P


def get_lambda(r, Ne):
    # waiting times ~ rates
    R = 2 * Ne * r
    lam = np.array(
        [1 + 2 * R, 3 + R, 6, 1 + R, 1 + R, 3, 3, 1, 1]
    )
    return lam


def evolve_P(P, lam):
    # state indices shifted down 1
    states = np.arange(9)

    path = []
    times = []

    scales = 1 / lam

    state = 0
    while state != 8:
        t = np.random.exponential(scales[state])

        path.append(state)
        times.append(t)

        state = np.random.choice(states, p=P[state])
    path.append(state)
    return np.array(path), np.array(times)


def get_T(path, times):
    # get T_l, T_r
    if 3 in path or 5 in path:
        # entered {3, 5}; right tree coalesced first
        coal_idx_r = np.where((path == 3) | (path == 5))[0][0]
        T_r = times[:coal_idx_r].sum()
        coal_idx_l = np.where((path == 7) | (path == 8))[0][0]
        T_l = times[:coal_idx_l].sum()
    elif 4 in path or 6 in path:
        # entered {4, 6}; left tree coalesced first
        coal_idx_l = np.where((path == 4) | (path == 6))[0][0]
        T_l = times[:coal_idx_l].sum()
        coal_idx_r = np.where((path == 7) | (path == 8))[0][0]
        T_r = times[:coal_idx_r].sum()
    else:
        # went straight to 8
        coal_idx = np.where(path == 8)[0][0]
        T_l = T_r = times[:coal_idx].sum()
    return T_l, T_r


"""
Attempt to write a very simple SMC algorithm for the two sample case
"""


def two_sample_SMC(rho):
    # time scaled by 2Ne. genome coordinates on [0, 1]
    T = np.random.exponential()
    x0 = 0
    x1 = np.random.exponential(2 / (rho * T))
    T_vec = [T]
    x_vec = [x0]
    while x1 < 1:
        u = np.random.uniform(0, T)
        s = u + np.random.exponential()
        T = s
        T_vec.append(T)
        x_vec.append(x1)
        x0 = x1
        x1 = x0 + np.random.exponential(2 / (rho * T))
    if x1 > 1:
        x1 = 1
    x_vec.append(x1)
    return np.array(x_vec), np.array(T_vec)


def __get_expected_TxTy(x, T, rho, rho_bins):
    #

    _x = rho * x
    lengths = x[1:] - x[:-1]
    centers = _x[:-1] + (x[1:] - x[:-1]) / 2
    b = len(rho_bins) - 1
    n = len(T)

    # get joint length sums for bins
    bin_norms = np.zeros(b)
    for i in np.arange(n):
        idx = np.searchsorted(rho_bins, centers[i + 1:])
        joint_l = lengths[i] * lengths[i + 1:]
        for k in np.arange(b):
            bin_norms[k] += joint_l[idx == k].sum()

    # get TxTy
    TxTy = np.zeros(b)
    for i in np.arange(n):
        idx = np.searchsorted(rho_bins, centers[i + 1:])
        joint_T = (T[i] * T[i + 1:]) * (lengths[i] * lengths[i + 1:])
        for k in np.arange(b):
            TxTy[k] += joint_T[idx == k].sum()

    return TxTy, bin_norms


"""
Equilibrium expectation of H2
"""


def compute_cov_tx_ty(Ne, r):
    # equilibrium. in coalescent units
    rho = 4 * Ne * r
    cov_tx_ty = (18 + rho) / (18 + 13 * rho + np.square(rho))
    return cov_tx_ty


def compute_eq_H2(Ne, mu, r):
    # E[H2] = E[2*mu*t_x * 2*mu*t_y]
    theta = 4 * Ne * mu
    cov_tx_ty = compute_cov_tx_ty(Ne, r)
    H2 = theta ** 2 * (1 + cov_tx_ty)
    return H2


if __name__ == "__main__":
    plt.rcParams['figure.dpi'] = 100
    matplotlib.use('Qt5Agg')
