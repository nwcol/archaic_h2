"""

"""
import demes
import numpy as np
import moments
from moments.Demes import Inference
import scipy
import pickle

from h2py import util, h2_parsing, theory
from h2py.h2stats_mod import H2stats


"""
optimization functions
"""


_out_of_bounds = -1e10
_n_calls = 0


_init_u = 1.3e-8
_lower_u = 1e-8
_upper_u = 1.6e-8


def optimize(
    objective_func,
    graph_file,
    param_file,
    data,
    u=None,
    method='NelderMead',
    max_iter=500,
    verbose=1,
    extra_args=None,
    out_file=None,
    perturb=0
):
    """
    Fit a graph defined in `graph_file` and parameterized by `param_file`
    to `data` using `objective_func` using a scipy optimization routine.
    """
    print(
        util.get_time(), f'fitting {objective_func.__name__} ' 
        f'to data for demes {data["pop_ids"]}'
    )
    builder = Inference._get_demes_dict(graph_file)
    options = Inference._get_params_dict(param_file)
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

    if out_file is not None: demes.dump(graph, out_file)
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



def fit_H2(*args, include_H=False, **kwargs):
    """
     
    """
    extra_args = dict(
        include_H=include_H
    )
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
    global _n_calls; _n_calls += 1

    if u is None: u = p[-1]

    if check_params(p, lower_bounds, upper_bounds, constraints) != 0:
        return -_out_of_bounds
    
    builder = Inference._update_builder(builder, options, p)
    graph = demes.Graph.fromdict(builder)
    model = moments_H2(graph, u=u, data=data)

    include_H = extra_args['include_H']
    ll = compute_ll(model, data, include_H=include_H)
    
    if verbose > 0 and _n_calls % verbose == 0:
        print_status(_n_calls, ll, p)

    return -ll


"""
Computing expected statistics
"""


def moments_H2(
    graph,
    data=None,
    rhos=None,
    theta=None,
    rs=None,
    bins=None,
    u=None,
    approximation='trapezoid',
    sampled_demes=None,
    sample_times=None
):
    """
    Compute expected H2 using moments.LD.

    If the approximation method is not None, then rs is treated as an array of
    bin edges; else it is treated as an array of points.
    """
    methods = ['midpoint', 'trapezoid', 'Simpsons', None]
    if approximation not in methods: 
        raise ValueError('approximation method is not recognized')

    if u is None and theta is None:
        raise ValueError('argument `u` or `theta` must be provided')

    if isinstance(graph, str):
        graph = demes.load(graph)

    if data is not None:
        sampled_demes = data['pop_ids']
        bins = data['bins']
    else:
        if sampled_demes is not None:
            graph_demes = [d.name for d in graph.demes]
            for d in sampled_demes:
                if d not in graph_demes: 
                    raise ValueError(f'deme {d} is not present in graph!')
        else:
            sampled_demes = [d.name for d in graph.demes if d.end_time == 0]

    if sample_times is None:
        end_times = {d.name: d.end_time for d in graph.demes}
        sample_times = [end_times[pop] for pop in sampled_demes]
    else:
        assert len(sample_times) == len(sampled_demes)

    if rs is None:
        if bins is None:
            bins = h2_parsing._default_bins
        rs = get_rs(bins, approximation)

    ld_stats = moments.Demes.LD(
        graph,
        sampled_demes,
        sample_times=sample_times,
        rho=rhos,
        theta=theta,
        r=rs,
        u=u
    )

    num_demes = len(sampled_demes)
    indices = [(i, j) for i in range(num_demes) for j in range(i, num_demes)]
    raw_H2 = np.zeros((len(rs), len(indices)))

    for k, (i, j) in enumerate(indices):
        phasing = True if i == j else False
        raw_H2[:, k] = ld_stats.H2(i, j, phased=phasing)

    H2 = approximate_H2(raw_H2, approximation)
    H2H = np.vstack((H2, ld_stats.H()))

    model = {
        'means': H2H,
        'pop_ids': sampled_demes,
        'bins': bins
    }
    return model


def get_rs(bins, approximation):
    """
    Get the r or rho values at which to evaluate H2 for a given approximation
    scheme.
    """
    key = (str(bins), approximation)
    if key in _rs_cache:
        rs = _rs_cache[key]

    elif approximation is None:
        rs = bins

    elif approximation == 'midpoint':
        rs = bins[:-1] + (bins[1:] - bins[:-1]) / 2

    elif approximation == 'trapezoid':
        rs = bins
    
    elif approximation == 'Simpsons':
        midpoints = (bins[1:] - bins[:-1]) / 2
        rs = np.sort(np.concatenate((bins, bins[:-1] + midpoints)))

    else:
        raise ValueError('unrecognized approximation method')

    return rs


def approximate_H2(raw, approximation):
    """
    
    """
    if approximation is None:
        ret = raw

    elif approximation == 'midpoint':
        ret = raw

    elif approximation == 'trapezoid':
        ret = 1/2 * (raw[:-1] + raw[1:])
    
    elif approximation == 'Simpsons':
        ret = 1/6 * raw[:-1:2] + 2/3 * raw[1::2] + 1/6 * raw[2::2]

    else:
        raise ValueError('unrecognized approximation method')

    return ret


def load_H2(file, graph=None):
    """
    
    """
    with open(file, 'rb') as fin:
        dic = pickle.load(fin)
    _data = dic[next(iter(dic))]
    if graph is not None:
        data = h2_parsing.subset_H2(_data, graph=graph)
    return data


"""
Computing log-likelihoods
"""


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
    xs = data['means']
    if data['covs'] is None:
        raise ValueError('data has no covariance matrix!')
    else:
        covs = data['covs']
    mus = model['means']
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


def compute_uncerts(
    graph_file,
    param_file,
    data,
    bootstrap_reps=None,
    u=None,
    delta=0.01,
    method='GIM'
):
    """
    
    """
    builder = Inference._get_demes_dict(graph_file)
    options = Inference._get_params_dict(param_file)
    pnames, p0, _, __ = Inference._set_up_params_and_bounds(options, builder)

    def model_func(p):

        nonlocal builder
        nonlocal options
        nonlocal data
        nonlocal u

        builder = Inference._update_builder(builder, options, p)
        graph = demes.Graph.fromdict(builder)
        model = moments_H2(graph, u=u, data=data)
        return model

    if method == 'FIM':
        H = compute_godambe(
            p0,
            model_func,
            data,
            bootstrap_reps=None,
            delta=delta,
            just_H=True
        )
        uncerts = np.sqrt(np.diag(np.linalg.inv(H)))

    elif method == 'GIM':
        if bootstrap_reps is None:
            raise ValueError('You need bootstrap_reps to use the GIM method!')
        G = compute_godambe(
            p0,
            model_func,
            data,
            bootstrap_reps,
            delta=delta
        )
        uncerts = np.sqrt(np.diag(np.linalg.inv(G)))
    else:
        uncerts = None

    return pnames, p0, uncerts


def compute_godambe(
    p0,
    model_func,
    data,
    bootstrap_reps,
    delta=0.01,
    just_H=False
):
    """
    Compute the Godambe matrix.
    """
    def ll_func(p, data):
        # compute log-likelihood given parameters, data
        # cache check
        key = tuple(p)
        if key in _model_cache:
            model = _model_cache[key]
        else:
            model = model_func(p)
            _model_cache[key] = model
        return compute_ll(model, data)

    H = -compute_hessian(p0, ll_func, data, delta=delta)

    if just_H:
        return H

    J = np.zeros((len(p0), len(p0)))
    for rep in bootstrap_reps:
        cU = compute_gradient(p0, ll_func, data, delta=delta)
        cJ = np.matmul(cU, cU.T)
        J += cJ

    J = J / len(bootstrap_reps)
    J_inv = np.linalg.inv(J)
    godambe = np.matmul(np.matmul(H,  J_inv), H)
    return godambe


def compute_hessian(p0, ll_func, data, delta=0.01):
    """
    
    """
    f0 = ll_func(p0, data)
    hs = delta * p0
    hessian = np.zeros((len(p0), len(p0)), dtype=np.float64)

    for i in range(len(p0)):
        for j in range(i, len(p0)):
            p = np.array(p0, copy=True, dtype=np.float64)

            if i == j:
                p[i] = p0[i] + hs[i]
                fp = ll_func(p, data)

                p[i] = p0[i] - hs[i]
                fm = ll_func(p, data)

                element = (fp - 2 * f0 + fm) / hs[i] ** 2

            else:
                p[i] = p0[i] + hs[i]
                p[j] = p0[j] + hs[j]
                fpp = ll_func(p, data)

                p[i] = p0[i] + hs[i]
                p[j] = p0[j] - hs[j]
                fpm = ll_func(p, data)

                p[i] = p0[i] - hs[i]
                p[j] = p0[j] + hs[j]
                fmp = ll_func(p, data)

                p[i] = p0[i] - hs[i]
                p[j] = p0[j] - hs[j]
                fmm = ll_func(p, data)

                element = (fpp - fpm - fmp + fmm) / (4 * hs[i] * hs[j])

            hessian[i, j] = element
            hessian[j, i] = element

    return hessian


def compute_gradient(p0, ll_func, data, delta=0.01):
    """
    
    """
    hs = delta * p0
    gradient = np.zeros((len(p0), 1))

    for i in range(len(p0)):
        p = np.array(p0, copy=True, dtype=float)

        p[i] = p0[i] + hs[i]
        fp = ll_func(p, data)

        p[i] = p0[i] - hs[i]
        fm = ll_func(p, data)
        gradient[i] = (fp - fm) / (2 * hs[i])

    return gradient


# caches
_rs_cache = {}
_model_cache = {}
_inv_cov_cache = {}


"""

"""


def estimate_rho_theta(stats, bins):
    """
    
    """
    def logistic_func(x, l, k, x0, y0):

        return y0 + l / (1 + np.exp(-k * (x - x0)))

    H = stats[-1]
    H2 = stats[:-1]
    midpoints = bins[:-1] + np.diff(bins) / 2
    log_midpoints = np.log10(midpoints)
    p0 = np.array([H ** 2, -1, -4, H ** 2])

    p_opt, p_cov = scipy.optimize.curve_fit(
        logistic_func,
        log_midpoints,
        H2,
        p0=p0
    )

    l, k, x0, y0 = p_opt

    theta_H2 = y0 ** 0.5
    rho_Ne = 1 / (4 * 10 ** -x0)
    ldfac = l / y0

    return p_opt

