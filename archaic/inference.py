
"""
Functions for computing approx composite likelihoods and inferring demographies
"""

import demes
import demesdraw
import numpy as np
import scipy
import matplotlib.pyplot as plt
from matplotlib import cm
import moments
import moments.Demes.Inference as m_infer
import time
from archaic import utils


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

    t0 = time.time()
    samples, pairs, means, covs = data
    # repackage data
    inv_covs = [np.linalg.inv(cov) for cov in covs]
    data = (samples, pairs, means, inv_covs)
    # get graph, parameter dicts
    builder = m_infer._get_demes_dict(graph_file_name)
    options = m_infer._get_params_dict(params_file_name)
    # get parameter bounds, constraints
    param_names, params_0, lower_bound, upper_bound = \
        m_infer._set_up_params_and_bounds(options, builder)
    constraints = m_infer._set_up_constraints(options, param_names)
    r = get_r(r_bins, method=approx_method)
    printout(0, 0, param_names, mode='s')
    # tuple of arguments to objective_fxn
    opt_args = (
        builder,
        data,
        options,
        r,
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
        params_opt, fopt, iters, funcalls, warnflag = out

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
        iters = out.nit
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
        iters = out.nit
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
        params_opt, fopt, iters, funcalls, warnflag = out

    else:
        return 1

    # build output graph using optimized parameters
    builder = m_infer._update_builder(builder, options, params_opt)
    graph = demes.Graph.fromdict(builder)
    global counter
    counter = 0
    out = {
        "param_names": param_names,
        "params_opt": params_opt,
        "fopt": fopt,
        "iter": iters,
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
        r,
        u,
        lower_bound=None,
        upper_bound=None,
        constraints=None,
        approx_method="simpsons",
        verbose=True,
        use_H=True
):

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
        builder = m_infer._update_builder(builder, options, params)
        graph = demes.Graph.fromdict(builder)
        # evaluate likelihood!
        log_lik = eval_log_lik(
            graph, data, r, u=u, approx_method=approx_method, use_H=use_H
        )
    if verbose > 0 and counter % verbose == 0:
        printout(counter, log_lik, params)
    return -log_lik


def printout(i, log_lik, params, mode='g'):
    # used to print parameter names and parameter values
    if mode == 'g':
        param_str = "array([%s])" % (", ".join(["%- 10g" % v for v in params]))
    elif mode == 's':
        param_str = "array([%s])" % (", ".join(["%- 10s" % v for v in params]))
    else:
        param_str = ""
    print(utils.get_time(), "%-8i, %-8g, %s" % (i, log_lik, param_str))


def eval_log_lik(
        graph,
        data,
        r_bins,
        u=1.35e-8,
        approx_method="simpsons",
        use_H=True
):

    samples, pairs, H2, inv_cov = data
    E_H2 = get_moments_stats(
        graph, samples, pairs, r_bins, u=u, approx_method=approx_method,
        get_H=use_H
    )
    E_H2 = [row for row in E_H2]
    n = len(H2)
    lik = sum([normal_log_lik(E_H2[i], inv_cov[i], H2[i]) for i in range(n)])
    return lik


def normal_log_lik(mu, inv_cov, x):
    # requires that the covariance matrix is already inverted
    log_lik = - (x - mu) @ inv_cov @ (x - mu)
    return log_lik


"""
Getting expected statistics using moments.LD
"""


def get_moments_stats(graph, samples, pairs, r, u=1.35e-8, get_H=True,
                      approx_method="simpsons"):

    n_samples = len(samples)
    ld_stats = moments.LD.LDstats.from_demes(
        graph, sampled_demes=samples, theta=None, r=r, u=u
    )
    H2 = np.array(
        [ld_stats.H2(samples, phased=True) for samples in samples] +
        [ld_stats.H2(x, y, phased=False) for x, y in pairs]
    ).T
    H2 = approximate_H2(H2, method=approx_method)
    if get_H:
        idx_pairs = utils.get_pair_idxs(n_samples)
        H = np.array(
            [ld_stats.H(pops=[i])[0] for i in range(n_samples)] +
            [ld_stats.H(pops=pair)[1] for pair in idx_pairs]
        )
        H2 = np.vstack([H2, H])
    return H2


"""
Numerical approximations
"""


approx_methods = [
    "left_bound",
    "right_bound",
    "midpoint",
    "simpsons"
]


def get_r(r_bins, method="simpsons"):
    # get values of r as required by approximation method
    if method not in approx_methods:
        raise ValueError(f"{method} is not in methods: {approx_methods}")
    if method == "left":
        r = r_bins[:-1]
    elif method == "right":
        r = r_bins[1:]
    elif method == "midpoint":
        r = r_bins[:-1] + np.diff(r_bins) / 2
    elif method == "simpsons":
        midpoints = r_bins[:-1] + np.diff(r_bins) / 2
        r = np.sort(np.concatenate([r_bins, midpoints]))
    else:
        r = None
    return r


def approximate_H2(arr, method="simpsons"):

    if method not in approx_methods:
        raise ValueError(f"{method} is not in methods: {approx_methods}")
    if method == "left":
        out = arr
    elif method == "right":
        out = arr
    # elif method == "midpoint":
    #    out = (arr[1:] + arr[:-1]) / 2
    elif method == "midpoint":
        out = arr
    elif method == "simpsons":
        n_rows = len(arr)
        n_bins = (n_rows - 1) // 2
        out = np.zeros((n_bins, arr.shape[1]))
        for i in range(n_bins):
            out[i] = 1 / 6 * (arr[i * 2] + 4 * arr[i * 2 + 1] + arr[i * 2 + 2])
    else:
        out = None
    return out


"""
Least squares optimization
"""


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
        builder = m_infer._update_builder(builder, options, params)
        graph = demes.Graph.fromdict(builder)
        s = eval_LS(
            graph, data, r_bins, u=u, approx_method=approx_method
        )
    if verbose > 0 and counter % verbose == 0:
        printout(counter, s, params)
    return s


def eval_LS(graph, data, r_bins, u=1.35e-8, approx_method="simpsons",
            use_H=True):

    sample_demes, H2, inv_cov = data
    E_H2 = get_moments_stats(
        graph, sample_demes, r_bins, u=u, approx_method="right", get_H=use_H
    )
    s = np.sum(np.square(H2 - E_H2))
    return s


"""
Plotting results of optimization
"""


def plot(graph, data, r_bins, u=1.35e-8, approx_method="simpsons", ci=1.96,
         two_sample=True, plot_demog=True, log_scale=False, use_H=True):
    # it's expected that the data tuple has H in it
    one_sample_cm = cm.gnuplot
    two_sample_cm = cm.terrain
    samples, pairs, means, covs = data
    n_samples = len(samples)
    n_pairs = len(pairs)
    # retrieve empirical statistics and confidences
    H2 = np.array(means[:-1])
    H = means[-1]
    idx = np.arange(n_samples + n_pairs)
    stds = np.sqrt(np.array([cov[idx, idx] for cov in covs]))
    H_err = stds[-1] * ci
    H2_err = stds[:-1] * ci
    # compute expected statistics and log likelihood
    r = get_r(r_bins, method="midpoint")
    _r = get_r(r_bins, method=approx_method)
    E_H_stats = get_moments_stats(
        graph, samples, pairs, r, u=u, approx_method="midpoint"
    )
    E_H2 = E_H_stats[:-1]
    E_H = E_H_stats[-1]
    if use_H:
        _data = (samples, pairs, means, np.linalg.inv(covs))
    else:
        _data = (samples, pairs, means[:-1], np.linalg.inv(covs[:-1]))
    log_lik = eval_log_lik(
        graph, _data, _r, u=u, approx_method=approx_method, use_H=use_H
    )
    colors = list(one_sample_cm(np.linspace(0, 0.95, n_samples)))
    if plot_demog:
        fig, axs = plt.subplot_mosaic(
            [[0, 1], [2, 2]], figsize=(10, 8), layout="constrained"
        )
        ax0 = axs[0]
        ax1 = axs[1]
        ax2 = axs[2]
        color_map = {name: colors[i] for i, name in enumerate(samples)}
        plot_graph(ax0, graph, color_map)
    else:
        fig, (ax1, ax2) = plt.subplots(
            1, 2, figsize=(12, 6), layout="constrained", width_ratios=[1.5, 1]
        )
    if two_sample:
        colors += list(two_sample_cm(np.linspace(0, 0.90, n_pairs)))
        samples += pairs
    plot_H(ax1, H, H_err, E_H, samples, colors)
    plot_H2(ax2, r, H2, H2_err, E_H2, samples, colors)
    fig.legend(fontsize=10, loc='center left', bbox_to_anchor=(1, 0.5))
    if log_scale:
        ax2.set_yscale("log")
    else:
        ax2.set_ylim(0, )
    fig.suptitle(f"log lik: {log_lik:.2e}")


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


"""
Loading and setting up bootstrap statistics
"""


def rename_data_samples(data, name_map):
    # name map of form old: new
    _names, _pairs, means, covs = data
    names = []
    pairs = []
    for _name in _names:
        if _name in name_map:
            names.append(name_map[_name])
        else:
            names.append(_name)
    for _pair in _pairs:
        pair = []
        for _name in _pair:
            if _name not in names:
                pair.append(name_map[_name])
            else:
                pair.append(_name)
        pairs.append(tuple(pair))
    return names, pairs, means, covs


def read_data(file_name, samples, get_H=True):
    # read bootstrap statistics from a .npz archive.
    archive = np.load(file_name)
    r_bins = archive["r_bins"]
    pairs = utils.get_pairs(samples)
    all_names = list(archive["sample_names"]) + list(archive["sample_pairs"])
    idx = np.array(
        [all_names.index(sample) for sample in samples]
        + [all_names.index(f"{x},{y}") for x, y in pairs]
    )
    n_bins = archive["n_bins"]
    means = [archive[f"H2_bin{i}_mean"][idx] for i in range(n_bins)]
    mesh_idx = np.ix_(idx, idx)
    covs = [archive[f"H2_bin{i}_cov"][mesh_idx] for i in range(n_bins)]
    if get_H:
        means += [archive["H_mean"][idx]]
        covs += [archive["H_cov"][mesh_idx]]
    return r_bins, (samples, pairs, np.array(means), np.array(covs))


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