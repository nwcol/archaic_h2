
import demes
import numpy as np


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
