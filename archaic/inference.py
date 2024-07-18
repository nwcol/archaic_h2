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
from archaic.spectra import H2Spectrum


out_of_bounds_val = 1e10
counter = 0
opt_methods = ['NelderMead', 'Powell', 'BFGS', 'LBFGSB']


"""
Graph optimization -- inference
"""


def optimize_H2(
    graph_fname,
    params_fname,
    data,
    max_iter=500,
    opt_method='NelderMead',
    u=1.35e-8,
    verbosity=1,
    use_H=True
):
    #
    t0 = time.time()
    if not use_H and data.has_H:
        data = data.remove_H
    builder = minf._get_demes_dict(graph_fname)
    options = minf._get_params_dict(params_fname)
    param_names, params_0, lower_bounds, upper_bounds = \
        minf._set_up_params_and_bounds(options, builder)
    constraints = minf._set_up_constraints(options, param_names)
    if verbosity > 0:
        ll_0 = compute_graph_file_ll_H2(graph_fname, data, u)
        printout(0, 0, param_names, mode='s')
        printout(0, ll_0, params_0)
    opt_args = (
        builder,
        options,
        data,
        u,
        lower_bounds,
        upper_bounds,
        constraints,
        verbosity,
    )
    if opt_method not in opt_methods:
        raise ValueError(f'method: {opt_method} is not in {opt_methods}')
    if opt_method == 'NelderMead':
        opt = scipy.optimize.fmin(
            objective_H2,
            params_0,
            args=opt_args,
            maxiter=max_iter,
            full_output=True
        )
        xopt, fopt, iters, func_calls, warnflag = opt
    elif opt_method == 'BFGS':
        opt = scipy.optimize.fmin_bfgs(
            objective_H2,
            params_0,
            args=opt_args,
            maxiter=max_iter,
            full_output=True
        )
        xopt, fopt, gopt, Bopt, func_calls, grad_calls, warnflag = opt
        iters = None
    elif opt_method == 'LBFGSB':
        opt = scipy.optimize.fmin_l_bfgs_b(
            objective_H2,
            params_0,
            args=opt_args,
            maxiter=max_iter,
            approx_grad=True
        )
        xopt, fopt, d = opt
        func_calls = d['funcalls']
        warnflag = d['warnflag']
        iters = d['nit']
    elif opt_method == 'Powell':
        opt = scipy.optimize.fmin_powell(
            objective_H2,
            params_0,
            args=opt_args,
            maxiter=max_iter,
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
        method=opt_method,
        fopt=-fopt,
        iters=iters,
        func_calls=func_calls,
        warnflag=warnflag
    )
    return graph, opt_info


def objective_H2(
    params,
    builder,
    options,
    data,
    u,
    lower_bounds=None,
    upper_bounds=None,
    constraints=None,
    verbosity=1
):
    # objective function for optimization with H2
    global counter
    counter += 1
    if lower_bounds is not None and np.any(params < lower_bounds):
        ll = -out_of_bounds_val
    elif upper_bounds is not None and np.any(params > upper_bounds):
        ll = -out_of_bounds_val
    elif constraints is not None and np.any(constraints(params) <= 0):
        ll = -out_of_bounds_val
    else:
        builder = minf._update_builder(builder, options, params)
        graph = demes.Graph.fromdict(builder)
        graph_data = H2Spectrum.from_graph(graph, data.sample_ids, data.r, u)
        ll = compute_ll_H2(graph_data, data)
    if verbosity > 0 and counter % verbosity == 0:
        printout(counter, ll, params)
    return -ll


def compute_ll_H2(model, data):
    # takes two H2Spectrum arguments
    ll = 0
    for i in range(data.n_bins):
        x = data.data[i]
        mu = model.data[i]
        inv_cov = data.inv_covs[i]
        ll += - (x - mu) @ inv_cov @ (x - mu)
    return ll


def compute_graph_file_ll_H2(graph_fname, data, u):
    #
    graph_data_0 = H2Spectrum.from_graph_file(
        graph_fname, data.sample_ids, data.r, u
    )
    ll = compute_ll_H2(graph_data_0, data)
    return ll


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
    print(f'time elapsed:\t{np.round(time.time() - t0, 2)} s\n')


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
Confidence intervals
"""


def compute_confidence():


    return 0


"""
Super-composite inference with H2 and the SFS
"""


def optimize_super_composite(
    graph_fname,
    params_fname,
    H2_data,
    SFS_data,
    L,
    max_iter=500,
    opt_method='NelderMead',
    u=1.35e-8,
    verbosity=1
):
    #
    t0 = time.time()

    sample_config = {sample: 2 for sample in SFS_data.pop_ids}

    if H2_data.has_H:
        H2_data = H2_data.remove_H()

    builder = minf._get_demes_dict(graph_fname)
    options = minf._get_params_dict(params_fname)
    param_names, params_0, lower_bounds, upper_bounds = \
        minf._set_up_params_and_bounds(options, builder)
    constraints = minf._set_up_constraints(options, param_names)

    if verbosity > 0:
        graph = demes.load(graph_fname)
        H2_model = H2Spectrum.from_graph(
            graph, H2_data.sample_ids, H2_data.r, u
        )
        SFS_model = moments.Demes.SFS(graph, samples=sample_config, u=u) * L
        lik0 = compute_ll_composite(H2_data, H2_model, SFS_data, SFS_model)
        printout(0, 0, param_names, mode='s')
        printout(0, lik0, params_0)

    opt_args = (
        builder,
        options,
        H2_data,
        SFS_data,
        sample_config,
        u,
        L,
        lower_bounds,
        upper_bounds,
        constraints,
        verbosity,
    )

    if opt_method not in opt_methods:
        raise ValueError(f'method: {opt_method} is not in {opt_methods}')
    if opt_method == 'NelderMead':
        opt = scipy.optimize.fmin(
            objective_composite,
            params_0,
            args=opt_args,
            maxiter=max_iter,
            full_output=True
        )
        xopt, fopt, iters, func_calls, warnflag = opt
    elif opt_method == 'BFGS':
        opt = scipy.optimize.fmin_bfgs(
            objective_composite,
            params_0,
            args=opt_args,
            maxiter=max_iter,
            full_output=True
        )
        xopt, fopt, gopt, Bopt, func_calls, grad_calls, warnflag = opt
        iters = None
    elif opt_method == 'LBFGSB':
        opt = scipy.optimize.fmin_l_bfgs_b(
            objective_composite,
            params_0,
            args=opt_args,
            maxiter=max_iter,
            approx_grad=True
        )
        xopt, fopt, d = opt
        func_calls = d['funcalls']
        warnflag = d['warnflag']
        iters = d['nit']
    elif opt_method == 'Powell':
        opt = scipy.optimize.fmin_powell(
            objective_composite,
            params_0,
            args=opt_args,
            maxiter=max_iter,
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
        method=opt_method,
        fopt=-fopt,
        iters=iters,
        func_calls=func_calls,
        warnflag=warnflag
    )
    return graph, opt_info


def objective_composite(
    params,
    builder,
    options,
    H2_data,
    SFS_data,
    sample_config,
    u,
    L,
    lower_bound=None,
    upper_bound=None,
    constraints=None,
    verbosity=1,
):
    global counter
    counter += 1
    if lower_bound is not None and np.any(params < lower_bound):
        ll = -out_of_bounds_val
    elif upper_bound is not None and np.any(params > upper_bound):
        ll = -out_of_bounds_val
    elif constraints is not None and np.any(constraints(params) <= 0):
        ll = -out_of_bounds_val
    else:
        builder = minf._update_builder(builder, options, params)
        graph = demes.Graph.fromdict(builder)
        H2_model = H2Spectrum.from_graph(
            graph, H2_data.sample_ids, H2_data.r, u
        )
        SFS_model = moments.Demes.SFS(graph, samples=sample_config, u=u) * L
        ll = compute_ll_composite(H2_data, H2_model, SFS_data, SFS_model)
    if verbosity > 0 and counter % verbosity == 0:
        printout(counter, ll, params)
    return -ll


def compute_ll_composite(H2_data, H2_model, SFS_data, SFS_model):
    #
    ll_H2 = compute_ll_H2(H2_model, H2_data)
    ll_SFS = moments.Inference.ll(SFS_model, SFS_data)
    ll = ll_H2 + ll_SFS
    return ll


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


def parse_graph_params(params_fname, graph_fnames, permissive=False):
    # shape (n_files, n_parameters)
    params = minf._get_params_dict(params_fname)
    if permissive:
        for i in range(len(params['parameters'])):
            params['parameters'][i]['lower_bound'] *= 0.99
            params['parameters'][i]['upper_bound'] *= 1.01
    names = None
    arr = []
    for fname in graph_fnames:
        g = minf._get_demes_dict(fname)
        names, vals, _, __, = minf._set_up_params_and_bounds(params, g)
        arr.append(vals)
    return names, np.array(arr)
