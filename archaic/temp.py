
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


def get_H(archive, sample_idx):

    H = archive["H_counts"][sample_idx].sum() / archive["site_counts"].sum()
    return H


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


def get_proper_H_squared(archive, sample_idx):
    # exclude cross-centromere pairings; eg (h_0 + h_1) / (s_0 + s_1)
    chroms = archive["chroms"]
    all_windows = archive["windows"]
    all_h = archive["H_counts"][sample_idx]
    all_s = archive["site_counts"]
    H = []
    for c in np.unique(chroms):
        ind = chroms == c
        h = all_h[ind]
        s = all_s[ind]
        windows = all_windows[ind]
        bounds = get_right_bounds(windows)
        # no centromere indicated by windows
        if np.all(bounds == bounds[0]):
            H.append(
                n_choose_2(h.sum()) / n_choose_2(s.sum())
            )
        # centromere
        else:
            h_0 = h[bounds == bounds[0]].sum()
            h_1 = h[bounds != bounds[0]].sum()
            s_0 = s[bounds == bounds[0]].sum()
            s_1 = s[bounds != bounds[0]].sum()
            H.append(
                (n_choose_2(h_0) + n_choose_2(h_1)) /
                (n_choose_2(s_0) + n_choose_2(s_1))
            )
    return np.array(H)


def window_discontinuity(windows):
    # previous function
    n_windows = len(windows)
    indicator = np.full(n_windows, 0)
    for i in range(n_windows - 1):
        if windows[i, 1] < windows[i+1, 0]:
            indicator[i] = 1
    indicator[-1] = 1
    return indicator


def _get_right_bounds(windows):
    # kind of messy
    n_windows = len(windows)
    bounds = np.zeros(n_windows)
    # detect whether there is a break in window coverage
    centromere = np.zeros(n_windows)
    for i in range(n_windows - 1):
        if windows[i, 1] < windows[i+1, 0]:
            centromere[i] = 1
    if np.any(centromere):
        for i in np.nonzero(centromere)[0]:
            bounds[:i + 1] = windows[i + 1, 1]
        bounds[i + 1:] = windows[-1, 1]
    else:
        bounds[:] = windows[-1, 1]
    return bounds


def get_right_bounds(windows):

    n_windows = len(windows)
    upper = windows[-1, 1]
    bounds = np.full(n_windows, upper)
    for i in range(n_windows - 1):
        if windows[i, 1] < windows[i+1, 0]:
            bounds[:i + 1] = windows[i, 1]
    return bounds


"""
Vectorizing pair counting
"""


def count_site_pairs(map_vals, r_bins, positions=None, window=None,
                     vectorized=False, bp_thresh=0, upper_bound=None):
    if bp_thresh:
        if not np.any(positions):
            raise ValueError("You must provide positions to use bp_thresh!")
    d_bins = two_locus.map_function(r_bins)
    if np.any(window):
        if not np.any(positions):
            raise ValueError("You must provide positions to use a window!")
        l_start, l_stop = one_locus.get_window_bounds(window, positions)
        if upper_bound:
            r_stop = np.searchsorted(positions, upper_bound)
        else:
            max_d = map_vals[l_stop - 1] + d_bins[-1]
            r_stop = np.searchsorted(map_vals, max_d)
        map_vals = map_vals[l_start:r_stop]
        if bp_thresh:
            positions = positions[l_start:r_stop]
    else:
        l_start = 0
        l_stop = len(map_vals)
    n_left_loci = l_stop - l_start
    cum_counts = np.zeros(len(d_bins), dtype=np.int64)
    if vectorized:
        edges = map_vals[:n_left_loci, np.newaxis] + d_bins[np.newaxis, :]
        counts = np.searchsorted(map_vals, edges)
        cum_counts = counts.sum(0)
        pair_counts = np.diff(cum_counts)
        # correction on lowest bin
        if r_bins[0] == 0:
            n_redundant = np.sum(
                np.arange(n_left_loci)
                - np.searchsorted(map_vals, map_vals[:n_left_loci])
            ) + n_left_loci
            pair_counts[0] -= n_redundant
    else:
        for i in np.arange(n_left_loci):
            if bp_thresh > 0:
                j = np.searchsorted(positions, positions[i] + bp_thresh + 1)
            else:
                j = i + 1
            _bins = d_bins + map_vals[i]
            cum_counts += np.searchsorted(map_vals[j:], _bins)
            if i % 1e6 == 0:
                print(f"locus {i} of {n_left_loci} loci")
        pair_counts = np.diff(cum_counts)
    return pair_counts



