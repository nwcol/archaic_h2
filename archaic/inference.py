"""
Functions for computing approx composite likelihoods and inferring demographies
"""


import demes
import numpy as np
import scipy
import moments
import moments.Demes.Inference as minf
import time
from archaic import utils


out_of_bounds_val = 1e10
counter = 0
opt_methods = ['NelderMead', 'Powell', 'BFGS', 'LBFGSB']
num_methods = [ "Simpsons", "midpoint"]


def optimize_with_H2(
    graph_fname,
    params_fname,
    data,
    r_bins,
    max_iters=1_000,
    opt_method='NelderMead',
    num_method="Simpsons",
    u=1.35e-8,
    verbosity=1,
    use_H=True,
    use_H2=True
):
    #
    t0 = time.time()
    samples, pairs, H, H_cov, H2, H2_cov = data
    data = (samples, pairs, H, np.linalg.inv(H_cov), H2, np.linalg.inv(H2_cov))
    builder = minf._get_demes_dict(graph_fname)
    options = minf._get_params_dict(params_fname)
    param_names, params_0, lower_bounds, upper_bounds = \
        minf._set_up_params_and_bounds(options, builder)
    constraints = minf._set_up_constraints(options, param_names)
    r = get_r(r_bins, method=num_method)
    if verbosity > 0:
        lik0 = eval_log_lik(demes.load(graph_fname), data, r, u=u)
        printout(0, 0, param_names, mode='s')
        printout(0, lik0, params_0)
    opt_args = (
        builder,
        data,
        options,
        r,
        u,
        lower_bounds,
        upper_bounds,
        constraints,
        num_method,
        verbosity,
        use_H,
        use_H2
    )
    if opt_method not in opt_methods:
        raise ValueError(f'method: {opt_method} is not in {opt_methods}')
    if opt_method == 'NelderMead':
        opt = scipy.optimize.fmin(
            objective_fxn,
            params_0,
            args=opt_args,
            maxiter=max_iters,
            full_output=True
        )
        xopt, fopt, iters, func_calls, warnflag = opt
    elif opt_method == 'BFGS':
        opt = scipy.optimize.fmin_bfgs(
            objective_fxn,
            params_0,
            args=opt_args,
            maxiter=max_iters,
            full_output=True
        )
        xopt, fopt, gopt, Bopt, func_calls, grad_calls, warnflag = opt
        iters = None
    elif opt_method == 'LBFGSB':
        opt = scipy.optimize.fmin_l_bfgs_b(
            objective_fxn,
            params_0,
            args=opt_args,
            maxiter=max_iters
        )
        xopt, fopt, d = opt
        func_calls = d['funcalls']
        warnflag = d['warnflag']
        iters = d['nit']
    elif opt_method == 'Powell':
        opt = scipy.optimize.fmin_powell(
            objective_fxn,
            params_0,
            args=opt_args,
            maxiter=max_iters,
            full_output=True
        )
        xopt, fopt, direc, iters, func_calls, warnflag = opt
    else:
        return 1
    builder = minf._update_builder(builder, options, xopt)
    graph = demes.Graph.fromdict(builder)
    global counter
    counter = 0
    end_printout(
        t0,
        fopt=fopt,
        iters=iters,
        func_calls=func_calls,
        warnflag=warnflag
    )
    opt_info = dict(
        fopt=fopt,
        iters=iters,
        func_calls=func_calls,
        warnflag=warnflag
    )
    return graph, opt_info


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
        num_method="Simpsons",
        verbosity=1,
        use_H=True,
        use_H2=True
):

    global counter
    counter += 1
    if lower_bound is not None and np.any(params < lower_bound):
        log_lik = -out_of_bounds_val
    elif upper_bound is not None and np.any(params > upper_bound):
        log_lik = -out_of_bounds_val
    elif constraints is not None and np.any(constraints(params) <= 0):
        log_lik = -out_of_bounds_val
    else:
        builder = minf._update_builder(builder, options, params)
        graph = demes.Graph.fromdict(builder)
        log_lik = eval_log_lik(
            graph,
            data,
            r,
            u,
            num_method=num_method,
            use_H=use_H,
            use_H2=use_H2
        )
    if verbosity > 0 and counter % verbosity == 0:
        printout(counter, log_lik, params)
    return -log_lik


def printout(i, log_lik, params, mode='g'):
    # used to print parameter names and parameter values
    if mode == 'g':
        param_str = ', '.join(['%- 10g' % v for v in params])
    elif mode == 's':
        param_str = ', '.join(['%- 10s' % v for v in params])
    else:
        param_str = ''
    print(utils.get_time(), '%-8i, %-8g, %s' % (i, log_lik, param_str))


def end_printout(t0, **kwargs):

    for key in kwargs:
        print(f'{key}:\t{kwargs[key]}')
    print(f'time elapsed:\t{np.round(time.time() - t0, 2)} s')


def eval_log_lik(
    graph,
    data,
    r,
    u,
    num_method="Simpsons",
    use_H=True,
    use_H2=True
):

    samples, pairs, H, H_cov, H2, H2_cov = data
    E_H, E_H2 = get_H_stats(
        graph, samples, pairs, r, u, num_method=num_method
    )
    lik = 0
    if use_H:
        lik += normal_log_lik(E_H, H_cov, H)
    if use_H2:
        n = len(E_H2)
        for i in range(n):
            lik += normal_log_lik(E_H2[i], H2_cov[i], H2[i])
    return lik


def normal_log_lik(mu, inv_cov, x):
    # requires that the covariance matrix is already inverted
    log_lik = - (x - mu) @ inv_cov @ (x - mu)
    return log_lik


"""
Getting expected statistics using moments.LD
"""


def get_H_stats(graph, samples, pairs, r, u, num_method="Simpsons"):
    # E_H has shape (n_samples), E_H2 has shape (n_bins, n_samples)
    ld_stats = moments.LD.LDstats.from_demes(
        graph, sampled_demes=samples, theta=None, r=r, u=u
    )
    n = len(samples)
    idx_pairs = utils.get_pair_idxs(n)
    E_H = np.array(
        [ld_stats.H(pops=[i])[0] for i in range(n)] +
        [ld_stats.H(pops=pair)[1] for pair in idx_pairs]
    )
    E_H2 = np.array(
        [ld_stats.H2(sample, phased=True) for sample in samples] +
        [ld_stats.H2(x, y, phased=False) for x, y in pairs]
    ).T
    E_H2 = approximate_H2(E_H2, method=num_method)
    if E_H2.ndim == 1:
        E_H2 = E_H2[:, np.newaxis]
    return E_H, E_H2


"""
Numerical approximations
"""


def get_r(r_bins, method='Simpsons'):
    #
    if method not in num_methods:
        raise ValueError(f'{method} is not in methods: {num_methods}')
    elif method == 'midpoint':
        r = r_bins[:-1] + np.diff(r_bins) / 2
    elif method == 'Simpsons':
        n = len(r_bins)
        r = np.zeros(n * 2 - 1)
        r[np.arange(n) * 2] = r_bins
        r[np.arange(n - 1) * 2 + 1] = r_bins[:-1] + np.diff(r_bins) / 2
    else:
        r = None
    return r


def approximate_H2(arr, method='Simpsons'):
    #
    if method not in num_methods:
        raise ValueError(f'{method} is not in methods: {num_methods}')
    elif method == 'midpoint':
        H2 = arr
    elif method == 'Simpsons':
        n = len(arr)
        b = (n - 1) // 2
        H2 = (
            1/6 * arr[np.arange(b) * 2]
            + 4/6 * arr[np.arange(b) * 2 + 1]
            + 1/6 * arr[np.arange(b) * 2 + 2]
        )
    else:
        H2 = None
    return H2


"""
Loading and setting up bootstrap statistics
"""


def scan_names(graph_fname, boot_fname):
    # return a list of sample names that are also deme names
    graph = demes.load(graph_fname)
    deme_names = [deme.name for deme in graph.demes]
    sample_names = [str(x) for x in np.load(boot_fname)["sample_names"]]
    names = []
    for name in sample_names:
        if name in deme_names:
            names.append(name)
    return names


def read_data(fname, sample_names):

    archive = np.load(fname)
    r_bins = archive["r_bins"]
    pairs = utils.get_pairs(sample_names)
    pair_names = utils.get_pair_names(sample_names)
    all_names = list(archive["sample_names"]) + list(archive["pair_names"])
    idx = np.array(
        [all_names.index(x) for x in sample_names]
        + [all_names.index(x) for x in pair_names]
    )
    mesh_idx = np.ix_(idx, idx)
    H = archive["H_mean"][idx]
    H_cov = archive["H_cov"][mesh_idx]
    H2 = archive["H2_mean"][:, idx]
    H2_cov = np.array([x[mesh_idx] for x in archive["H2_cov"]])
    return r_bins, (sample_names, pairs, H, H_cov, H2, H2_cov)


"""
Inference with the SFS
"""


def optimize_with_SFS(
    data_fname,
    graph_fname,
    params_fname,
    out_fname,
    u,
    L,
    names,
    method="fmin",
    max_iter=1_000,
    verbosity=0,
):
    #
    archive = np.load(data_fname)
    sample_names = archive["sample_names"]
    idx = utils.get_pairs(sample_names).index(tuple(names))
    sfs = archive["sfs"][idx]
    data = moments.Spectrum(sfs, pop_ids=names)
    uL = u * L
    fit = minf.optimize(
        graph_fname,
        params_fname,
        data,
        maxiter=max_iter,
        uL=uL,
        output=out_fname,
        verbose=verbosity,
        method=method,
        overwrite=True
    )
    print(fit)
    return 0


"""
Graph permutation
"""


def log_uniform(lower, upper):
    # sample parameters log-uniformly
    log_lower = np.log10(lower)
    log_upper = np.log10(upper)
    log_draws = np.random.uniform(log_lower, log_upper)
    draws = 10 ** log_draws
    return draws


def permute_graph(graph_fname, param_fname, out_fname):
    # uniformly and randomly pick parameter values
    builder = minf._get_demes_dict(graph_fname)
    param_dict = minf._get_params_dict(param_fname)
    param_names, params0, lower_bounds, upper_bounds = \
        minf._set_up_params_and_bounds(param_dict, builder)
    if np.any(np.isinf(upper_bounds)):
        raise ValueError("all upper bounds must be specified!")
    constraints = minf._set_up_constraints(param_dict, param_names)
    above1 = np.where(lower_bounds >= 1)[0]
    below1 = np.where(lower_bounds < 1)[0]
    n = len(params0)
    satisfied = False
    params = None
    while not satisfied:
        params = np.zeros(n)
        params[above1] = np.random.uniform(
            lower_bounds[above1], upper_bounds[above1]
        )
        params[below1] = log_uniform(
            lower_bounds[below1], upper_bounds[below1]
        )
        if constraints:
            if np.all(constraints(params) > 0):
                satisfied = True
        else:
            satisfied = True
    builder = minf._update_builder(builder, param_dict, params)
    graph = demes.Graph.fromdict(builder)
    demes.dump(graph, out_fname)


"""
Parse parameters from large number of graphs
"""


def parse_graph_params(params_fname, graph_fnames):
    #
    params = minf._get_params_dict(params_fname)
    names = None
    arr = []
    for fname in graph_fnames:
        g = minf._get_demes_dict(fname)
        names, vals, _, __, = minf._set_up_params_and_bounds(params, g)
        arr.append(vals)
    return names, np.array(arr)
