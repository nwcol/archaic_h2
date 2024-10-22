"""

"""
import demes
import numpy as np
import moments
from moments.Demes import Inference
import scipy

from h2py import util, theory
from h2py.h2stats_mod import H2stats


"""
optimization functions
"""


_out_of_bounds = -1e10
_n_calls = 0


_init_u = 1.3e-8
_lower_u = 1e-8
_upper_u = 1.6e-8


def fit_H2(*args, include_H=False, **kwargs):
    """
    Fit 
    """
    extra_args = dict(include_H=include_H)
    ret = optimize(
        objective_H2,
        *args, 
        extra_args=extra_args, 
        **kwargs
    )
    return ret


def objective_H2(
    p,
    builder,
    options,
    data,
    u=None,
    lower_bounds=None,
    upper_bounds=None,
    constraints=None,
    verbose=None,
    extra_args=None
):
    """
    Evalute the log likelihood of parameters `p` against empirical H2
    statistics.
    """
    global _n_calls
    _n_calls += 1
    if u is None: u = p[-1]
    if check_params(p, lower_bounds, upper_bounds, constraints) != 0:
        return -_out_of_bounds
    builder = Inference._update_builder(builder, options, p)
    graph = demes.Graph.fromdict(builder)
    model = H2stats.from_demes(graph, u=u, template=data)
    include_H = extra_args['include_H']
    ll = compute_ll(model, data, include_H=include_H)
    if verbose > 0 and _n_calls % verbose == 0:
        print_status(_n_calls, ll, p)
    return -ll


def optimize(
    objective_func,
    graph_fname,
    param_fname,
    data,
    u=None,
    method='NelderMead',
    max_iter=500,
    verbose=1,
    extra_args=None,
    out_fname=None,
    perturb=0
):
    """
    Fit a graph defined in `graph_fname` and parameterized by `param_fname`
    to `data` using `objective_func` using a scipy optimization routine.
    """
    print(
        util.get_time(), f'fitting {objective_func.__name__} ' 
        f'to data for demes {data.pop_ids}'
    )
    builder = Inference._get_demes_dict(graph_fname)
    options = Inference._get_params_dict(param_fname)
    pnames, p0, lower_bounds, upper_bounds = \
        Inference._set_up_params_and_bounds(options, builder)
    constraints = Inference._set_up_constraints(options, pnames)

    if u is None:
        print(util.get_time(), f'fitting u as a free parameter')
        pnames = np.append(pnames, 'u')
        p0 = np.append(p0, _init_u)
        lower_bounds = np.append(lower_bounds, _lower_u)
        upper_bounds = np.append(upper_bounds, _upper_u)
        
    if perturb > 0: 
        p0 = Inference._perturb_params_constrained(
            p0, 
            perturb, 
            lower_bound=lower_bounds, 
            upper_bound=upper_bounds,
            cons=constraints
        )
    print_start(pnames, p0)

    args = (
        builder,
        options,
        data,
        u,
        lower_bounds,
        upper_bounds,
        constraints,
        verbose,
        extra_args
    )
    
    methods = ['NelderMead', 'Powell', 'BFGS', 'LBFGSB']
    if method not in methods:
        raise ValueError(f'method: {method} is not in {methods}')
    
    if method == 'NelderMead':
        opt = scipy.optimize.fmin(
            objective_func,
            p0,
            args=args,
            maxiter=max_iter,
            full_output=True
        )
        p = opt[0]
        fopt, num_iter, func_calls, flag = opt[1:5]

    elif method == 'BFGS':
        opt = scipy.optimize.fmin_bfgs(
            objective_func,
            p0,
            args=args,
            maxiter=max_iter,
            full_output=True
        )
        p = opt[0]
        fopt, _, __, func_calls, grad_calls, flag = opt[1:7]
        # is it correct to equate these?
        num_iter = grad_calls

    elif method == 'LBFGSB':
        _bounds = list(zip(lower_bounds, upper_bounds))
        opt = scipy.optimize.fmin_l_bfgs_b(
            objective_func,
            p0,
            args=args,
            maxiter=max_iter,
            bounds=_bounds,
            epsilon=1e-2,
            pgtol=1e-5,
            approx_grad=True
        )
        p, fopt, d = opt
        num_iter = d['nit']
        func_calls = d['funcalls']
        flag = d['warnflag']

    elif method == 'Powell':
        opt = scipy.optimize.fmin_powell(
            objective_func,
            p0,
            args=args,
            maxiter=max_iter,
            full_output=True,
        )
        p = opt[0]
        fopt, _, num_iter, func_calls, flag = opt[1:6]

    else:
        return 1

    global _n_calls
    print_status(_n_calls, 'fit p:', p)

    if u is None:
        u_fitted = True
        _u = p[:-1]
    else:
        u_fitted = False
        _u = u
    info = dict(
        method=method,
        objective_func=objective_func.__name__,
        fopt=-fopt,
        max_iter=max_iter,
        num_iter=num_iter,
        func_calls=func_calls,
        flag=flag,
        u_fitted=u_fitted,
        u=_u
    )
    print('\n'.join([f'{key}: {info[key]}' for key in info]))
    if builder is None or options is None:
        return p, info
    builder = Inference._update_builder(builder, options, p)
    graph = demes.Graph.fromdict(builder)
    graph.metadata['opt_info'] = info

    if out_fname is not None: demes.dump(graph, out_fname)
    else: return graph


def print_start(pnames, p0):
    """
    Print out parameter names and initial values.
    """
    print_status(0, 'pnames', pnames)
    print_status(0, 'p0', p0)
    return


def print_status(n_calls, ll, p):
    """
    Print the number of function calls, the log-likelihood, and the current 
    parameter values.
    """
    t = util.get_time()
    _n = f'{n_calls:<4}'
    if isinstance(ll, float):
        _ll = f'{np.round(ll, 2):>10}'
    else:
        _ll = f'{ll:>10}'
    fmt_p = []
    for x in p:
        if isinstance(x, str):
            fmt_p.append(f'{x:>10}')
        else:
            if x > 1:
                fmt_p.append(f'{np.round(x, 1):>10}')
            else:
                sci = np.format_float_scientific(x, 2, trim='k')
                fmt_p.append(f'{sci:>10}')
    _p = ''.join(fmt_p)
    print(t, _n, _ll, '[', _p, ']')


def check_params(p, lower_bounds, upper_bounds, constraints):
    """
    Check whether any parameters violate bounds or constraints, returning 1
    if they do and 0 otherwise.
    """
    ret = 0
    if lower_bounds is not None and np.any(p < lower_bounds):
        ret = 1
    elif upper_bounds is not None and np.any(p > upper_bounds):
        ret = 1
    elif constraints is not None and np.any(constraints(p) <= 0):
        ret = 1
    return ret


"""
Computing log-likelihoods
"""


_inv_cov_cache = {}


def compute_ll(model, data, include_H=False):
    """
    Compute the log-likelihood of expected against empirical H2, where both
    are stored as H2stats instances. If `use_H` is False, then the H matrix is
    excluded from the likelihood computation. 
    """
    return compute_bin_ll(model, data, include_H=include_H).sum()


def compute_bin_ll(model, data, include_H=False):
    """
    Get log-likelihood per bin from H2stats instances `model` and `data`. 
    If use_H is True, then likelihood is also computed for H.
    """
    xs = data.stats
    if data.covs is None:
        raise ValueError('data has no covariance matrix!')
    else:
        covs = data.covs
    mus = model.stats
    if include_H:
        bin_ll = _compute_bin_ll(xs, mus, covs)
    else:
        bin_ll = _compute_bin_ll(xs[:-1], mus[:-1], covs[:-1])
    return bin_ll


def _compute_ll(xs, mus, covs):
    """
    Compute log-likelihood over bare arrays of coordinates, means and
    variance-covariances matrices.
    """
    return _compute_bin_ll(xs, mus, covs).sum()


def _compute_bin_ll(xs, mus, covs):
    """
    Compute log-likelihood per bin over arrays of coordinates, means and 
    covariance matrices. Uses caching to increase speed.
    """
    lens = np.array([len(xs), len(mus), len(covs)])
    if not np.all(lens == lens[0]):
        raise ValueError('xs, mus, covs lengths do not match')
    bin_ll = np.zeros(len(xs))
    for i in range(len(xs)):
        if i in _inv_cov_cache and np.all(_inv_cov_cache[i]['cov'] == covs[i]):
            inv_cov = _inv_cov_cache[i]['inv']
        else:
            inv_cov = np.linalg.inv(covs[i])
            _inv_cov_cache[i] = dict(cov=covs[i], inv=inv_cov)
        bin_ll[i] = log_gaussian(xs[i], mus[i], inv_cov)
    return bin_ll


def log_gaussian(x, mu, inv_cov):
    """
    Compute the log of the multivariate gaussian law with means `mu` and
    pre-inverted variance-covariance matrix `inv_cov` at values `x`, 
    ignoring the coefficient. 
    """
    return -np.matmul(np.matmul(x - mu, inv_cov), x - mu)


"""
Computing confidence intervals
"""


def get_uncerts(
    graph_fname,
    options_fname,
    data,
    bootstraps=None,
    u=None,
    delta=0.01,
    method='GIM'
):
    """
    
    """
    builder = Inference._get_demes_dict(graph_fname)
    options = Inference._get_params_dict(options_fname)
    pnames, p0, _, __ = Inference._set_up_params_and_bounds(options, builder)

    if u is None:
        # my temporary means of getting mutation rate into p0
        g = demes.load(graph_fname)
        _u = float(g.metadata['opt_info']['u'])
        pnames.append('u')
        p0 = np.append(p0, _u)
        fit_u = True
    else:
        fit_u = False

    def model_func(p):
        # takes parameters and returns expected statistics
        nonlocal builder
        nonlocal options
        nonlocal data
        nonlocal u
        nonlocal fit_u

        if fit_u:
            u = p[-1]

        builder = Inference._update_builder(builder, options, p)
        graph = demes.Graph.fromdict(builder)
        model = H2Spectrum.from_graph(graph, data.sample_ids, data.r, u)
        return model

    if method == 'FIM':
        H = get_godambe_matrix(
            model_func,
            p0,
            data,
            bootstraps,
            delta,
            just_H=True
        )
        uncerts = np.sqrt(np.diag(np.linalg.inv(H)))

    elif method == 'GIM':
        if bootstraps is None:
            raise ValueError('You need bootstraps to use the GIM method!')
        godambe_matrix = get_godambe_matrix(
            model_func,
            p0,
            data,
            bootstraps,
            delta
        )
        uncerts = np.sqrt(np.diag(np.linalg.inv(godambe_matrix)))
    else:
        uncerts = None

    return pnames, p0, uncerts


_ll_cache = {}


def get_godambe_matrix(
    model_func,
    p0,
    data,
    bootstraps,
    delta,
    just_H=False
):
    """
    
    """
    def func(p, data):
        # compute log-likelihood given parameters, data
        # cache check
        key = tuple(p)
        if key in _ll_cache:
            model = _ll_cache[key]
        else:
            model = model_func(p)
            _ll_cache[key] = model
        return get_ll(model, data)

    H = -get_hessian(func, p0, data, delta)

    if just_H:
        return H

    J = np.zeros((len(p0), len(p0)))
    for i, bootstrap in enumerate(bootstraps):
        cU = get_gradient(func, p0, delta, bootstrap)
        cJ = cU @ cU.T
        J += cJ

    J = J / len(bootstraps)
    J_inv = np.linalg.inv(J)
    godambe_matrix = H @ J_inv @ H
    return godambe_matrix


def get_hessian(ll_func, p0, data, delta):
    """
    
    """
    f0 = ll_func(p0, data)
    hs = delta * p0

    hessian = np.zeros((len(p0), len(p0)))

    for i in range(len(p0)):
        for j in range(i, len(p0)):
            _p = np.array(p0, copy=True, dtype=float)

            if i == j:
                _p[i] = p0[i] + hs[i]
                fp = ll_func(_p, data)
                _p[i] = p0[i] - hs[i]
                fm = ll_func(_p, data)

                element = (fp - 2 * f0 + fm) / hs[i] ** 2

            else:
                _p[i] = p0[i] + hs[i]
                _p[j] = p0[j] + hs[j]
                fpp = ll_func(_p, data)

                _p[i] = p0[i] + hs[i]
                _p[j] = p0[j] - hs[j]
                fpm = ll_func(_p, data)

                _p[i] = p0[i] - hs[i]
                _p[j] = p0[j] + hs[j]
                fmp = ll_func(_p, data)

                _p[i] = p0[i] - hs[i]
                _p[j] = p0[j] - hs[j]
                fmm = ll_func(_p, data)

                element = (fpp - fpm - fmp + fmm) / (4 * hs[i] * hs[j])

            hessian[i, j] = element
            hessian[j, i] = element

    return hessian


def get_gradient(func, p0, delta, args):
    """
    
    """
    # should be changed to match moments version
    hs = delta * p0

    # column
    gradient = np.zeros((len(p0), 1))

    for i in range(len(p0)):
        _p = np.array(p0, copy=True, dtype=float)

        _p[i] = p0[i] + hs[i]
        fp = func(_p, args)

        _p[i] = p0[i] - hs[i]
        fm = func(_p, args)
        gradient[i] = (fp - fm) / (2 * hs[i])

    return gradient
