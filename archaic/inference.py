
"""
Functions for computing approx composite likelihoods and inferring demographies
"""

from datetime import datetime
import demes
import demesdraw
import numpy as np
import scipy
import matplotlib.pyplot as plt
from matplotlib import cm
import moments
import moments.Demes.Inference as infer
import time


out_of_bounds_val = 1e10
counter = 0


def optimize(
        graph_file_name,
        params_file_name,
        data,
        r_bins,
        max_iter=1_000,
        opt_method="fmin",
        approx_method="simpsons",
        u=1.35e-8,
        verbose=True,
        use_H=True
):
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
    :param verbose:
    :param use_H:
    :return:
    """
    t0 = time.time()

    # invert the covariance matrix
    sample_ids, means, covs = data
    inv_covs = [np.linalg.inv(cov) for cov in covs]
    data = (sample_ids, means, inv_covs)

    # get demes graph dictionary and parameter dictionary
    builder = infer._get_demes_dict(graph_file_name)
    options = infer._get_params_dict(params_file_name)

    # use existing moments functionality to get parameter bounds etc
    param_names, params_0, lower_bound, upper_bound = \
        infer._set_up_params_and_bounds(options, builder)
    constraints = infer._set_up_constraints(options, param_names)

    param_str = "array([%s])" % (", ".join(["%- 10s" % v for v in param_names]))
    print(get_time(), "%-8i, %-8g, %s" % (0, 0, param_str))

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
        approx_method,
        verbose,
        use_H
    )
    opt_methods = ["fmin", "BFGS", "LBFGSB", "LS"]
    if opt_method not in opt_methods:
        raise ValueError(f"method: {opt_method} is not in {opt_methods}")
    if opt_method == "fmin":
        out = scipy.optimize.fmin(
            objective_fxn,
            params_0,
            args=opt_args,
            disp=True,
            maxiter=max_iter,
            maxfun=max_iter,
            full_output=True,
        )
        params_opt, fopt, iter, funcalls, warnflag = out

    elif opt_method == "BFGS":
        out = scipy.optimize.minimize(
            objective_fxn,
            params_0,
            args=opt_args,
            method="BFGS",
            options={"maxiter": max_iter}
        )
        params_opt = out.x
        fopt = out.fun
        iter = out.nit
        funcalls = out.nfev
        warnflag = out.status

    elif opt_method == "LBFGSB":
        out = scipy.optimize.minimize(
            objective_fxn,
            params_0,
            args=opt_args,
            method="L-BFGS-B",
            options={"maxiter": max_iter}
        )
        params_opt = out.x
        fopt = out.fun
        iter = out.nit
        funcalls = out.nfev
        warnflag = out.status

    elif opt_method == "LS":
        out = scipy.optimize.fmin(
            LS_objective_fxn,
            params_0,
            args=opt_args,
            disp=True,
            maxiter=max_iter,
            maxfun=max_iter,
            full_output=True,
        )
        params_opt, fopt, iter, funcalls, warnflag = out

    else:
        return 1

    # build output graph using optimized parameters
    builder = infer._update_builder(builder, options, params_opt)
    graph = demes.Graph.fromdict(builder)

    global counter
    counter = 0

    out = {
        "param_names": param_names,
        "params_opt": params_opt,
        "fopt": fopt,
        "iter": iter,
        "funcalls": funcalls,
        "warnflag": warnflag,
        "time_elapsed": time.time() - t0
    }
    return graph, out


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
        approx_method="simpsons",
        verbose=True,
        use_H=True
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
    log_lik = None

    global counter
    counter += 1

    # bounds and constraints checks
    if lower_bound is not None and np.any(params < lower_bound):
        log_lik = -out_of_bounds_val
    elif upper_bound is not None and np.any(params > upper_bound):
        log_lik = -out_of_bounds_val
    elif constraints is not None and np.any(constraints(params) <= 0):
        log_lik = -out_of_bounds_val
    else:
        # update builder and build graph
        builder = infer._update_builder(builder, options, params)
        graph = demes.Graph.fromdict(builder)
        # evaluate likelihood!
        log_lik = eval_log_lik(
            graph, data, r_bins, u=u, approx_method=approx_method, use_H=use_H
        )
    if verbose > 0 and counter % verbose == 0:
        param_str = "array([%s])" % (", ".join(["%- 10g" % v for v in params]))
        print(get_time(), "%-8i, %-8g, %s" % (counter, log_lik, param_str))
    return -log_lik


def LS_objective_fxn(
        params,
        builder,
        data,
        options,
        r_bins,
        u,
        lower_bound=None,
        upper_bound=None,
        constraints=None,
        approx_method="simpsons",
        verbose=True,
        use_H=True
):
    s = None

    global counter
    counter += 1

    # bounds check
    if lower_bound is not None and np.any(params < lower_bound):
        s = out_of_bounds_val
    elif upper_bound is not None and np.any(params > upper_bound):
        s = out_of_bounds_val
    # constraints check
    elif constraints is not None and np.any(constraints(params) <= 0):
        s = out_of_bounds_val
    else:
        # update builder and build graph
        builder = infer._update_builder(builder, options, params)
        graph = demes.Graph.fromdict(builder)

        # evaluate likelihood!
        s = eval_LS(
            graph, data, r_bins, u=u, approx_method=approx_method
        )
    # print summary
    if verbose > 0 and counter % verbose == 0:
        param_str = "array([%s])" % (", ".join(["%- 12g" % v for v in params]))
        print(get_time(), "%-8i, %-10g, %s" % (counter, s, param_str))
    return s


def eval_LS(graph, data, r_bins, u=1.35e-8, approx_method="simpsons",
            use_H=True):

    sample_demes, H2, inv_cov = data
    E_H2 = get_two_locus_stats(
        graph, sample_demes, r_bins, u=u, approx_method="right", get_H=use_H
    )
    s = np.sum(np.square(H2 - E_H2))
    return s


def eval_log_lik(graph, data, r_bins, u=1.35e-8, approx_method="simpsons",
                 use_H=True):
    """
    Compute the composite likelihood of a demes graph defining a demography,
    given empirical means and covariances obtained from sequence data and a
    mutation rate u.

    :param graph:
    :param data:
    :param r_bins:
    :param u:
    :param approx_method:
    :param use_H:
    :return:
    """
    sample_demes, H2, inv_cov = data

    # compute expected H2 and H values given the demes graph
    E_H2 = get_two_locus_stats(
        graph, sample_demes, r_bins, u=u, approx_method=approx_method,
        get_H=use_H
    )
    E_H2 = [row for row in E_H2]

    # compute log likelihood with multivariate gaussian function
    n = len(H2)
    lik = 0
    for i in range(n):
        lik += pre_inverted_normal_log_lik(E_H2[i], inv_cov[i], H2[i])
    return lik


def get_two_locus_stats(graph, sample_demes, r_bins, u=1.35e-8,
                        approx_method="simpsons", get_H=True):
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
    n_samples = len(sample_demes)
    deme_pairs = enumerate_pairs(sample_demes)
    r = get_r_points(r_bins, method=approx_method)
    ld_stats = moments.LD.LDstats.from_demes(
        graph, sampled_demes=sample_demes, theta=None, r=r, u=u
    )
    # get H2 and approximate its value in each r bin
    H2 = np.array(
        [ld_stats.H2(sample_id, phased=True) for sample_id in sample_demes] +
        [ld_stats.H2(id_x, id_y, phased=False) for id_x, id_y in deme_pairs]
    ).T
    H2 = approximate_midpoints(H2, method=approx_method)
    idx_pairs = enumerate_pairs(np.arange(n_samples))
    if get_H:
        H = np.array(
            [ld_stats.H(pops=[i])[0] for i in range(n_samples)] +
            [ld_stats.H(pops=pair)[1] for pair in idx_pairs]
        )
        H2 = np.vstack([H2, H])
    return H2


def get_r_points(r_bins, method="simpsons"):
    """
    Return a vector of r-points for computing H2

    :param r_bins:
    :param method:
    :return:
    """
    methods = [None, "right", "midpoint", "simpsons"]
    if method not in methods:
        raise ValueError(f"{method} is not in methods: {methods}")
    if method == "right":
        r = r_bins[1:]
    elif method == "midpoint":
        r = r_bins[:-1] + np.diff(r_bins) / 2
    elif method == "simpsons":
        midpoints = r_bins[:-1] + np.diff(r_bins) / 2
        r = np.sort(np.concatenate([r_bins, midpoints]))
    else:
        return 1
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
    methods = ["right", "midpoint", "simpsons"]
    if method not in methods:
        raise ValueError(
            f"{method} is not in methods: {methods}"
        )
    if method == "right":
        out = arr
    elif method == "midpoint":
        out = (arr[1:] + arr[:-1]) / 2
    elif method == "simpsons":
        n_rows = len(arr)
        n_bins = (n_rows - 1) // 2
        out = np.zeros((n_bins, arr.shape[1]))
        for i in range(n_bins):
            out[i] = 1 / 6 * (arr[i * 2] + 4 * arr[i * 2 + 1] + arr[i * 2 + 2])
    else:
        return 1
    return out


def pre_inverted_normal_log_lik(mu, inv_cov, x):
    """

    :param mu: vector of means
    :param inv_cov: pre-inverted covariance matrix
    :param x: vector of points
    :return:
    """
    log_lik = - (x - mu) @ inv_cov @ (x - mu)
    return log_lik


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


"""
Visualizing results of optimization
"""


def plot(graph, data, r_bins, u=1.35e-8, approx_method="simpsons", ci=1.96,
         two_sample=True, plot_demog=True, log_scale=False, use_H=True):

    one_sample_cm = cm.gnuplot
    two_sample_cm = cm.terrain

    sample_names, means, covs = data
    pair_names = enumerate_pairs(sample_names)
    n_samples = len(sample_names)
    n_pairs = len(pair_names)

    # retrieve empirical statistics and confidences
    H2 = np.array(means[:-1])
    H = means[-1]
    idx = np.arange(n_samples + n_pairs)
    stds = np.sqrt(np.array([cov[idx, idx] for cov in covs]))
    H_err = stds[-1] * ci
    H2_err = stds[:-1] * ci

    # compute expected statistics and log likelihood
    E_H_stats = get_two_locus_stats(
        graph, sample_names, r_bins, u=u, approx_method=approx_method
    )
    E_H2 = E_H_stats[:-1]
    E_H = E_H_stats[-1]
    if use_H:
        _data = (sample_names, means, np.linalg.inv(covs))
    else:
        _data = (sample_names, means[:-1], np.linalg.inv(covs))
    log_lik = eval_log_lik(
        graph, _data, r_bins, u=u, approx_method=approx_method, use_H=use_H
    )
    r = get_r_points(r_bins, method="midpoint")
    colors = list(one_sample_cm(np.linspace(0, 0.95, n_samples)))
    if plot_demog:
        fig, axs = plt.subplot_mosaic(
            [["x", "y"], ["z", "z"]], figsize=(10, 8), layout="constrained"
        )
        ax2 = axs["x"]
        ax1 = axs["y"]
        ax0 = axs["z"]
        color_map = {name: colors[i] for i, name in enumerate(sample_names)}
        plot_graph(ax2, graph, color_map)
    else:
        fig, (ax0, ax1) = plt.subplots(
            1, 2, figsize=(12, 6), layout="constrained", width_ratios=[1.5, 1]
        )
    if two_sample:
        colors += list(two_sample_cm(np.linspace(0, 0.90, n_pairs)))
        sample_names += pair_names
    plot_H2(ax0, r, H2, H2_err, E_H2, sample_names, colors)
    plot_H(ax1, H, H_err, E_H, sample_names, colors)

    fig.legend(fontsize=9, loc='center left', bbox_to_anchor=(1, 0.5))

    if log_scale:
        ax0.set_yscale("log")
    else:
        ax0.set_ylim(0, )

    fig.suptitle(f"log lik: {log_lik:.2e}")
    return ax0, ax1


def plot_H2(ax, r, H2, H2_err, E_H2, names, colors):

    for i, name in enumerate(names):
        if type(name) == str:
            style = "solid"
        else:
            style = "dotted"
            name = f"{name[0]}-{name[1]}"
        ax.errorbar(
            r, H2[:, i], yerr=H2_err[:, i], color=colors[i], fmt=".",
            label=name, capsize=0
        )
        ax.plot(r, E_H2[:, i], color=colors[i], linestyle=style)
    ax.set_xscale("log")
    ax.set_ylabel("$H_2$")
    ax.set_xlabel("r")
    ax.grid(alpha=0.2)
    return ax


def plot_H(ax, H, H_err, E_H, names, colors):

    abbrev_names = []
    for i, name in enumerate(names):
        if type(name) == str:
            name = name[:3]
        else:
            name = f"{name[0][:3]}-{name[1][:3]}"
        abbrev_names.append(name)
        ax.errorbar(i, H[i], yerr=H_err[i], color=colors[i], fmt='.')
        ax.scatter(i, E_H[i], color=colors[i], marker='+')
    ax.set_ylim(0, )
    ax.set_ylabel("$H$")
    ax.set_xticks(np.arange(len(names)), abbrev_names)
    ax.grid(alpha=0.2)
    return ax


def plot_graph(ax, graph, color_map):

    demesdraw.tubes(graph, ax=ax, colours=color_map)
    return ax


def get_time():

    return " [" + datetime.strftime(datetime.now(), "%Y-%m-%d %H:%M:%S") + "]"


"""
Loading and setting up bootstrap statistics
"""


def rename_data_samples(data, name_map):
    # name map of form old: new
    _names, means, covs = data
    names = []
    for _name in _names:
        if _name in name_map:
            names.append(name_map[_name])
        else:
            names.append(_name)
    return names, means, covs


def read_data(file_name, sample_names, get_H=True):
    """
    Read bootstrap statistics from a .npz archive.

    :param file_name:
    :param sample_names:
    :param get_H: if True, read heterozygosities from the archive and append
        them to the data arrays
    :return:
    """
    archive = np.load(file_name)
    r_bins = archive["r_bins"]
    sample_pairs = enumerate_pairs(sample_names)
    pair_names = [f"{x},{y}" for (x, y) in sample_pairs]
    sample_names += pair_names
    all_names = list(archive["sample_names"]) + list(archive["sample_pairs"])
    idx = np.array(
        [all_names.index(sample) for sample in sample_names]
    )
    n_bins = archive["n_bins"]
    means = [archive[f"H2_bin{i}_mean"][idx] for i in range(n_bins)]
    mesh_idx = np.ix_(idx, idx)
    covs = [archive[f"H2_bin{i}_cov"][mesh_idx] for i in range(n_bins)]
    if get_H:
        means += [archive["H_mean"][idx]]
        covs += [archive["H_cov"][mesh_idx]]
    return r_bins, (sample_names, np.array(means), np.array(covs))


"""
Other statistical functions
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
