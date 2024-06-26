
import demes
import matplotlib.pyplot as plt
import matplotlib
import numpy as np


plt.rcParams['figure.dpi'] = 100
matplotlib.use('Qt5Agg')



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
H2
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























def get_deme(graph, deme_name):

    deme_names = [deme.name for deme in graph.demes]
    idx = deme_names.index(deme_name)
    return graph.demes[idx]


def get_ancestral_deme(graph, deme_name):



    return 0


def get_Ne_trajectory(graph, deme_name):

    focus_deme = get_deme(graph, deme_name)
    _focus_deme = focus_deme
    ancestral_demes = []
    ur_deme = None
    while not ur_deme:
        ancestral_deme = get_deme(graph, _focus_deme.ancestors[0])
        ancestral_demes.append(ancestral_deme)
        if len(ancestral_deme.ancestors) == 0:
            ur_deme = ancestral_deme
        else:
            _focus_deme = ancestral_deme


def estimate_H(graph, deme_name, mu=1.35e-8):
    # not robust to branched scenarios! Or much complexity at all

    return 0
