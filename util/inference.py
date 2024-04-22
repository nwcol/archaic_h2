
"""
Functions for computing approx composite likelihoods and inferring demographies
"""

import demes
import demesdraw
import numpy as np
import scipy
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import cm
import moments
import moments.Demes.Inference as moments_inference
import ruamel
from two_locus import r, r_edges
from util import file_util
from util import plots


out_of_bounds_val = -1e10


def optimize(graph_file_name, params_file_name, data, r_bins, max_iter=1_000,
             opt_method="fmin", approx_method="simpsons", u=1.35e-8):
    """
    Optimize a demographic model.

    :param graph_file_name: path to .yaml file specifying initial demes model
    :param params_file_name: path to .yaml options file defining parameters
    :param r_bins: ndarray specifying r bin edges
    :param data: tuple of (sample_ids, empirical means, empirical covariances)
    :param max_iter:
    :param opt_method:
    :param approx_method: method to approximate H2 within each r bin
    :param u: generational mutation rate
    :return:
    """

    # get demes graph dictionary and parameter dictionary
    builder = moments_inference._get_demes_dict(graph_file_name)
    options = moments_inference._get_params_dict(params_file_name)

    # use existing moments functionality to get parameter bounds etc
    param_names, params_0, lower_bound, upper_bound = \
        moments_inference._set_up_params_and_bounds(options, builder)
    constraints = moments_inference._set_up_constraints(options, param_names)

    # tuple of arguments to objective_fxn
    opt_args = (
        builder,
        data,
        options,
        r_bins,
        u,
        lower_bound,
        upper_bound,
        constraints,
        approx_method
    )

    # check to make sure opt_method is a valid choice
    opt_methods = ["fmin"]
    if opt_method not in opt_methods:
        raise ValueError(f"method: {opt_method} is not in {opt_methods}")

    # conduct the optimization according to selected method
    if opt_method == "fmin":
        out = scipy.optimize.fmin(
            objective_fxn,
            params_0,
            args=opt_args,
            disp=True,
            maxiter=max_iter,
            maxfun=max_iter,
            full_output=True
        )
        params_opt, fopt, iter, fun_calls, warn_flag = out

    # build output graph using optimized parameters
    builder = moments_inference._update_builder(builder, options, params_opt)
    graph = demes.Graph.fromdict(builder)

    return graph, param_names, params_opt, fopt, iter, fun_calls, warn_flag


def objective_fxn(
        params,
        builder,
        data,
        options,
        r_bins,
        u,
        lower_bound=None,
        upper_bound=None,
        constraints=None,
        approx_method="simpsons"
):
    """
    Check validity of parameters within bounds and constraints, then evaluate
    likelihood given a builder for a demes graph, a set of data (H, H2) in the
    form of empirical means and covariances, and a mutation rate.

    :param params:
    :param builder:
    :param data:
    :param options:
    :param r_bins:
    :param u: generational mutation rate
    :param lower_bound:
    :param upper_bound:
    :param constraints: function to check that parameter constraints are met
    :param approx_method:
    :return:
    """

    # bounds check
    if lower_bound is not None and np.any(params < lower_bound):
        return -out_of_bounds_val
    if upper_bound is not None and np.any(params > upper_bound):
        return -out_of_bounds_val

    # constraints check
    if constraints is not None and np.any(constraints(params) <= 0):
        return -out_of_bounds_val

    # update builder and build graph
    builder = moments_inference._update_builder(builder, options, params)
    graph = demes.Graph.fromdict(builder)

    # evaluate likelihood!
    log_lik = eval_log_lik(
        graph, data, r_bins, u=u, approx_method=approx_method
    )

    return -log_lik



def eval_log_lik(graph, data, r_bins, u=1.35e-8, approx_method="simpsons"):
    """
    Compute the composite likelihood of a demes graph defining a demography,
    given empirical means and covariances obtained from sequence data and a
    mutation rate u.

    :param graph:
    :param data:
    :param r_bins:
    :param u:
    :param approx_method:
    :return:
    """
    # unpack data
    sample_ids, emp_means, emp_covs = data

    # compute expected H2 and H values given the demes graph
    H2, H = get_two_locus_stats(
        graph, sample_ids, r_bins, u=u, approx_method=approx_method
    )
    expected_stats = [row for row in H2] + [H]

    # compute log likelihood with multivariate gaussian function
    n_components = len(emp_means)
    composite_lik = 0
    for i in range(n_components):
        lik = normal_log_lik(expected_stats[i], emp_covs[i], emp_means[i])
        composite_lik += lik

    return composite_lik


def get_two_locus_stats(graph, sample_ids, r_bins, u,
                        approx_method="simpsons"):
    """
    Get an array of expected statistics. Each array column corresponds to
    one r bin defined by r_bins except the last, which holds expected
    heterozygosities. The order of entry along rows is thus;

    sample_id_0, ... sample_id_n-1, sample_pair_0, ... sample_pair_n*(n-1)/2-1

    :param graph:
    :param sample_ids:
    :param r_bins:
    :param u:
    :param approx_method: method for approximating H values along the curve
    :return:
    """
    # map sample_ids to deme names
    sample_pairs = enumerate_pairs(sample_ids)

    # get points of r to compute H2 at, as required by approximation method
    r = find_r_points(r_bins, method=approx_method)

    n_samples = len(sample_ids)
    ld_stats = moments.LD.LDstats.from_demes(
        graph, sampled_demes=sample_ids, theta=None, r=r, u=u
    )

    # get H2 and approximate its value in the r bins
    H2 = np.array(
        [ld_stats.H2(sample_id, phased=True) for sample_id in sample_ids] +
        [ld_stats.H2(id_x, id_y, phased=False) for id_x, id_y in sample_pairs]
    ).T
    H2 = approximate_midpoints(H2, method=approx_method)
    idx_pairs = enumerate_pairs(np.arange(n_samples))

    # get H
    H = np.array(
        [ld_stats.H(pops=[i])[0] for i in range(n_samples)] +
        [ld_stats.H(pops=pair)[1] for pair in idx_pairs]
    )

    return H2, H


def find_r_points(r_bins, method="simpsons"):
    """
    Given a method for numerically approximating H2 values, return a vector
    of r values to compute H2 over.

    :param r_bins:
    :param method:
    :return:
    """
    methods = [None, "right", "midpoint", "simpsons"]
    if method not in methods:
        raise ValueError(
            f"{method} is not in methods: {methods}"
        )

    if method == "right":
        r = r_bins[1:]

    elif method == "midpoint":
        r = r_bins[:-1] + np.diff(r_bins) / 2

    elif method == "simpsons":
        midpoints = r_bins[:-1] + np.diff(r_bins) / 2
        r = np.sort(np.concatenate([r_bins, midpoints]))
    return r


def approximate_midpoints(arr, method="simpsons"):
    """
    Numerically approximate bin values across an array.

    Note: different array sizes will interact differently with different
    methods-

    :param arr:
    :param method:
    :return:
    """
    methods = ["simpsons"]
    if method not in methods:
        raise ValueError(
            f"{method} is not in methods: {methods}"
        )
    if method == "right":
        out_arr = arr

    elif method == "midpoint":
        out_arr = (arr[1:] + arr[:-1]) / 2

    elif method == "simpsons":
        n_rows = len(arr)
        n_bins = (n_rows - 1) // 2
        out_arr = np.zeros((n_bins, arr.shape[1]))
        for i in range(n_bins):
            out_arr[i] = 1 / 6 * (
                arr[i * 2]
                + 4 * arr[i * 2 + 1]
                + arr[i * 2 + 2]
            )
    return out_arr


def normal_log_lik(mu, cov, x):
    """

    :param mu: vector of means
    :param cov: covariance matrix
    :param x: vector of points
    :return:
    """
    log_lik = - (x - mu) @ np.linalg.inv(cov) @ (x - mu)
    return log_lik


def enumerate_pairs(items):
    """
    Return a list of sorted 2-tuples containing every pair of objects in
    'items'

    :param items: list of objects
    :return: list of 2-tuples of paired objects
    """
    n = len(items)
    pairs = []
    for i in np.arange(n):
        for j in np.arange(i + 1, n):
            pair = [items[i], items[j]]
            pair.sort()
            pairs.append((pair[0], pair[1]))
    return pairs


def subset_cov_matrix(cov_matrix, all_ids, subset_ids):
    """


    :param cov_matrix:
    :param all_ids:
    :param subset_ids:
    :return:
    """
    idx = np.array([all_ids.index(x) for x in subset_ids])
    mesh_idx = np.ix_(idx, idx)
    subset_matrix = cov_matrix[mesh_idx]
    return subset_matrix


"""
Visualizing results of optimization
"""


def plot(graph, data, r_bins, u=1.35e-8, approx_method="simpsons",
         plot_H=False, plot_one_sample=True, plot_two_sample=False):

    one_sample_cm = cm.gnuplot
    two_sample_cm = cm.terrain

    sample_ids, means, covs = data
    emp_H2 = np.array(means[:-1])
    emp_H = means[-1]
    sample_pairs = enumerate_pairs(sample_ids)
    n_samples = len(sample_ids)
    n_pairs = len(sample_pairs)

    # compute statistics and log likelihood
    exp_H2, exp_H = get_two_locus_stats(
        graph, sample_ids, r_bins, u=u, approx_method=approx_method
    )
    log_lik = eval_log_lik(
        graph, data, r_bins, u=u, approx_method=approx_method
    )
    x = find_r_points(r_bins, method="midpoint")

    fig, ax = plt.subplots(figsize=(7, 6), layout="constrained")

    if plot_one_sample:
        colors = one_sample_cm(np.linspace(0, 0.9, n_samples))

        for i, sample_id in enumerate(sample_ids):
            ax.plot(x, emp_H2[:, i], color=colors[i], label=sample_id)
            ax.scatter(x, exp_H2[:, i], color=colors[i], marker='x')

            if plot_H:
                ax.scatter(0.5, emp_H[i] ** 2, color=colors[i], marker='+')
                ax.scatter(0.5, exp_H[i] ** 2, color=colors[i], marker='x')

    if plot_two_sample:
        colors = two_sample_cm(np.linspace(0, 0.9, n_pairs))

        for i, pair in enumerate(sample_pairs):
            j = i + n_samples
            ax.plot(
                x, emp_H2[:, j], color=colors[i], label=pair, linestyle="dotted"
            )
            ax.scatter(x, exp_H2[:, j], color=colors[i], marker='1')

            if plot_H:
                ax.scatter(0.5, emp_H[j] ** 2, color=colors[i], marker='+')
                ax.scatter(0.5, exp_H[j] ** 2, color=colors[i], marker='1')

    ax.set_ylim(0, )
    ax.set_xscale("log")
    ax.set_ylabel("H_2")
    ax.set_xlabel("r bin")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.2)
    ax.set_title(f"log likelihood: {log_lik:.2e}")
    return ax


"""
Loading and setting up bootstrap statistics
"""


def load_(boot_path, sample_ids, idx_offset, n_bins):

    means = []
    covs = []
    for i in range(idx_offset, n_bins):
        means.append(
            read_boot_mean(f"{boot_path}/bin{i}_mean.txt", sample_ids)
        )
        covs.append(read_boot_cov(f"{boot_path}/bin{i}_cov.txt", sample_ids))
    means.append(read_boot_mean(f"{boot_path}/H_mean.txt", sample_ids))
    covs.append(read_boot_cov(f"{boot_path}/H_cov.txt", sample_ids))
    return means, covs


def read_boot_mean(file_name, sample_ids):

    arr = np.loadtxt(file_name)
    header = file_util.read_header(file_name)
    all_sample_ids = header["cols"]
    idx = np.array(
        [all_sample_ids.index(x) for x in sample_ids] +
        [all_sample_ids.index(x) for x in enumerate_pairs(sample_ids)]
    )
    return arr[idx]


def read_boot_cov(file_name, sample_ids):

    arr = np.loadtxt(file_name)
    header = file_util.read_header(file_name)
    all_sample_ids = header["cols"]
    idx = np.array(
        [all_sample_ids.index(x) for x in sample_ids] +
        [all_sample_ids.index(x) for x in enumerate_pairs(sample_ids)]
    )
    mesh_idx = np.ix_(idx, idx)
    return arr[mesh_idx]


def load_means(boot_path, sample_ids):

    boot_header = file_util.read_header(
        f"{boot_path}/bin4_mean.txt"
    )
    all_sample_ids = boot_header["cols"]
    raw_means = [
        np.loadtxt(f"{boot_path}/bin{i}_mean.txt") for i in range(4, 19)
    ]
    raw_means.append(
        np.loadtxt(f"{boot_path}/H_mean.txt")
    )
    idx = np.array(
        [all_sample_ids.index(x) for x in sample_ids] +
        [all_sample_ids.index(x) for x in enumerate_pairs(sample_ids)]
    )
    means = np.array([mean[idx] for mean in raw_means])
    means = [means[:, i] for i in range(means.shape[1])]
    return means


"""
Other inferential/statistics functions
"""


def permutation_test(A, B, n_resamplings):

    mean_A = A.mean()
    mean_B = B.mean()
    difference = mean_B - mean_A
    pooled = np.concatenate([A, B])
    n_A = len(A)
    differences = np.zeros(n_resamplings)
    for i in np.arange(n_resamplings):
        samples = np.random.choice(pooled, size=n_A, replace=False)
        differences[i] = mean_B - samples.mean()
    p = np.count_nonzero(differences >= difference) / n_resamplings
    return p
