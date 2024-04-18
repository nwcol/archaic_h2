
"""
Functions for computing approx. composite likelihood and inferring demographies
"""

import demes
import demesdraw
import numpy as np
import scipy
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import cm
import moments
from two_locus import r, r_edges
from util import file_util
from util import plotting


"""
Max-likelihood function and Moments inference

Required input for inference involving n samples in b r-bins:
    Obtained from bootstrap:
    1. properly-ordered list of:
        b identically-ordered empirical mean-H2 vectors of length (n + 1) 
            choose 2  
        1 mean-H vector of same length
    2. properly-ordered list of:
        b identically-ordered H2 covariance matrices of size (n + 1) choose 2 
        1 H covariance vector of same size
        
    Obtained from model:
    3. a .yaml demes file defining a demographic model, loaded as a demes graph
    4. a .yaml specification file parameterizing the demes file
    5. a mapping of sampled deme names to coordinates in the empirical arrays
"""


def eval_lik(graph, emp_means, emp_covs, r_bins, u=1.3e-8):
    """
    Compute the composite likelihood of a demes graph defining a demography,
    given empirical means and covariances obtained from sequence data and a
    mutation rate u.

    :param graph:
    :param emp_means:
    :param emp_covs:
    :param r_bins:
    :param u:
    :return:
    """
    expected = get_2_locus_stats(graph, sample_ids, r_bins, u=u)

    n_components = len(emp_means)
    composite_lik = 0
    for i in range(n_components):
        lik = normal_log_lik(expected[i], emp_covs[i], emp_means[i])
        composite_lik += lik
        print(i, lik)
    return composite_lik


def get_2_locus_stats(graph, name_map, sample_ids, sample_pairs, r_bins, u):
    """
    Get a list of vectors of expected statistics. Each vector corresponds to
    one r bin defined by r_bins except the last, which holds expected
    heterozygosities. In each vector, the order of entry is

    sample_id_0, ... sample_id_n-1, sample_pair_0, ... sample_pair_n*(n-1)/2-1

    :param graph:
    :param name_map:
    :param sample_ids:
    :param sample_pairs:
    :param r_bins:
    :param u:
    :return:
    """
    # map demes
    r = r_bins[1:]
    n_bins = len(r)
    n_samples = len(sample_ids)
    n_stats = n_samples + len(sample_pairs)
    ld_stats = moments.LD.LDstats.from_demes(
        graph, sampled_demes=sample_ids, theta=None, r=r, u=u
    )
    expected_H2 = np.array(
        [ld_stats.H2(sample_id, phased=True) for sample_id in sample_ids] +
        [ld_stats.H2(id_x, id_y, phased=False) for id_x, id_y in sample_pairs]
    )
    idx_pairs = enumerate_pairs(np.arange(n_stats))
    expected_H = np.array(
        [ld_stats.H(pops=[i])[0] for i in range(n_stats)] +
        [ld_stats.H(pops=pair)[1] for pair in idx_pairs]
    )
    expected_stats = [expected_H2[]]
    # figure out this shape better
    return expected_stats


def normal_log_lik(mu, cov, x):
    """

    :param mu:
    :param cov:
    :param x:
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





def plot(graph, sample_ids, r, idx_start=4, title=None, single=True,
         double=False):

    boot_path = "/home/nick/Projects/archaic/statistics/main/bootstrap"


    n_samples = len(sample_ids)
    n_pairs = int(n_samples * (n_samples - 1) / 2)
    colors = cm.gnuplot(np.linspace(0, 0.9, n_samples))
    pair_colors = cm.terrain(np.linspace(0, 0.9, n_pairs))

    est_H, est_H2, est_H2_xy = moments_H(graph, sample_ids, r_bins=r)
    lik = eval_lik(graph, sample_ids, r, boot_path, idx_start=idx_start)
    means = load_means(boot_path, sample_ids)
    emp_H2 = [mean[:-1] for mean in means]
    H = [mean[-1] for mean in means]

    fig, ax = plt.subplots(figsize=(7, 6), layout="constrained")
    if single:
        for i, label in enumerate(sample_ids):
            ax.plot(r, emp_H2[i], color=colors[i], label=label)
            ax.scatter(r, est_H2[i], color=colors[i], marker='x')

            ax.scatter(0.5, H[i] ** 2, color=colors[i], marker='+')
            ax.scatter(0.5, est_H[i] ** 2, color=colors[i], marker='x')

    if double:
        for i, pair in enumerate(enumerate_pairs(sample_ids)):
            ax.plot(r, emp_H2[i + n_samples], color=pair_colors[i], label=pair,
                    linestyle="dotted")
            ax.scatter(r, est_H2_xy[i], color=pair_colors[i], marker='1')

            ax.scatter(0.5, H[i + n_samples] ** 2, color=pair_colors[i], marker='+')
            ax.scatter(0.5, est_H[i + n_samples] ** 2, color=pair_colors[i], marker='1')

    ax.set_ylim(0, )
    ax.set_xscale("log")
    ax.set_ylabel("H_2")
    ax.set_xlabel("r bin")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.2)
    if title:
        ax.set_title(title + f"; log lik: {lik:.2e}")
    else:
        ax.set_title(f"log lik: {lik:.2e}")
    return ax


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


def moments_H(graph, sampled_demes, r_bins, u=1.5e-8):

    y = moments.LD.LDstats.from_demes(
        graph, sampled_demes=sampled_demes, r=r_bins, u=u, theta=None
    )
    n_demes = len(sampled_demes)
    H = [y.H(pops=[i])[0] for i in range(n_demes)]
    idx_pairs = enumerate_pairs(np.arange(n_demes))

    H += [y.H(pops=pair)[1] for pair in idx_pairs]
    H2 = [y.H2(sample_id, phased=True) for sample_id in sampled_demes]
    sample_pairs = enumerate_pairs(sampled_demes)
    H2_xy = [y.H2(id_x, id_y, phased=False) for id_x, id_y in sample_pairs]
    return H, H2, H2_xy














def dep_main(graph, sampled_demes, sample_times, sample_ids, r):

    path = "/home/nick/Projects/archaic/statistics/main/bootstrap"

    boot_header = file_util.read_header(
        f"{path}/bin4_mean.txt"
    )
    all_sample_ids = boot_header["cols"]
    raw_means = [
        np.loadtxt(f"{path}/bin{i}_mean.txt") for i in range(4, 19)
    ]
    raw_means.append(
        np.loadtxt(f"{path}/H_mean.txt")
    )
    raw_covs = [
        np.loadtxt(f"{path}/bin{i}_cov.txt") for i in range(4, 19)
    ]
    raw_covs.append(
        np.loadtxt(f"{path}/H_cov.txt")
    )
    idx = np.array(
        [all_sample_ids.index(x) for x in sample_ids] +
        [all_sample_ids.index(x) for x in enumerate_pairs(sample_ids)]
    )
    mesh_idx = np.ix_(idx, idx)
    means = [mean[idx] for mean in raw_means]
    covs = [cov[mesh_idx] for cov in raw_covs]

    # compute expected stats under demography graph
    est_H, est_H2, est_H2_xy = moments_H(
        graph, sampled_demes, sample_times, r_bins=r
    )
    est_H2 = est_H2 + est_H2_xy

    # transform expected stats into usable shape
    # for now, assume that sampled_demes and sample_ids are in the same order!
    n_bins = len(means) - 1
    n_samples = len(est_H2)
    est_means = [
        np.array(
            [est_H2[i][bin_idx] for i in range(n_samples)]
        ) for bin_idx in range(n_bins)
    ]
    est_means += np.array(est_H)

    log_lik = composite_likelihood(est_means, covs, means)

    return log_lik


def composite_likelihood(mus, covs, xs):
    """


    :param mus:
    :param covs:
    :param xs:
    :return:
    """
    n = len(mus)
    log_liks = []
    for i in range(n):
        log_liks.append(normal_log_lik(mus[i], covs[i], xs[i]))
    log_lik = sum(log_liks)
    return log_lik






def get_sample_times(graph, sample_demes):
    # probably not needed
    n_demes = len(graph.demes)
    deme_names = [graph.demes[i].name for i in range(n_demes)]
    deme_times = []
    deme_idx = [deme_names.index(sample_deme) for sample_deme in sample_demes]

















def multivariate_normal(mu, Sigma, x):
    """
    Evaluate a multivariate normal PDF with locations mu, covariances Sigma
    at x.

    :param mu:
    :param Sigma:
    :param x:
    :return:
    """
    k = len(mu)
    d = (
        ((2 * np.pi) ** k * np.linalg.det(Sigma)) ** -0.5
        * np.exp(-0.5 * (x - mu) @ np.linalg.inv(Sigma) @ (x - mu))
    )
    return d


def unscaled_multivariate_normal(mu, Sigma, x):
    """
    Evaluate a multivariate normal probability measure but do not scale it to
    be a pdf
    
    :param mu: 
    :param Sigma: 
    :param x: 
    :return: 
    """
    d = np.exp(-0.5 * (x - mu) @ np.linalg.inv(Sigma) @ (x - mu))
    return d


def log_f(mu, Sigma, x):
    f_theta = -0.5 * (x - mu) @ np.linalg.inv(Sigma) @ (x - mu)
    return f_theta


def lik(mu_dict, Sigma_dict, x_dict):

    # check to make sure dicts all have the same keys?
    liks = np.zeros(len(x_dict))
    for i, key in enumerate(x_dict):
        d = unscaled_multivariate_normal(
            mu_dict[key], Sigma_dict[key], x_dict[key]
        )
        liks[i] = - np.log(d)
    lik = liks.sum()
    return lik













def moments_test(graph):

    X = "X"
    sampled_demes = [X]
    sample_times = [0]

    r = [0, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1]
    u = 1.5e-8

    y = moments.LD.LDstats.from_demes(
        graph, sampled_demes=sampled_demes, sample_times=sample_times, r=r,
        u=u, theta=None
    )

    H2_X = y.H2(X, phased=True)

    fig = plt.figure(1)
    ax = plt.subplot(1, 1, 1)
    ax.plot(r, H2_X, label="X")
    ax.set_xscale("log")
    ax.legend()
    ax.set_xlabel("r")
    ax.set_ylabel("H2")
    fig.tight_layout()
    plt.show()
    return 0
    
    


















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





"""

Usage: python {yaml name} {first sampled population} {second sampled population}

For example: python one_pop.yaml X X


install with
pip install git+https://github.com/MomentsLD/moments.git@devel

import sys
import moments
import demes
import matplotlib.pylab as plt


graph = sys.argv[1]
X = sys.argv[2]
Y = sys.argv[3]

g = demes.load(graph)

if X == Y:
    sampled_demes = [X]
    sample_times = [0]

else:
    sample_demes = [X, Y]
    sample_times = [0, 0]

r = [0, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1]  # you'll set these as your bin edges
u = 1.5e-8
Ne = 10000

y = moments.LD.LDstats.from_demes(
    g, sampled_demes=sampled_demes, sample_times=sample_times, u=u, r=r, Ne=Ne
)

H2_X = y.H2(X, phased=True)
H2_Y = y.H2(Y, phased=True)
H2_XY = y.H2(X, Y, phased=False)

# these values are for the bin edges, not bin midpoint - but we can average...
H2_X = (H2_X[:-1] + H2_X[1:]) / 2
H2_Y = (H2_Y[:-1] + H2_Y[1:]) / 2
H2_XY = (H2_XY[:-1] + H2_XY[1:]) / 2
r_mids = [(r[i] + r[i + 1]) / 2 for i in range(len(r) - 1)]

fig = plt.figure(1)
ax = plt.subplot(1, 1, 1)
ax.plot(r_mids, H2_X, label="X")
ax.plot(r_mids, H2_Y, label="Y")
ax.plot(r_mids, H2_XY, label="XY")
ax.set_xscale("log")
ax.legend()
ax.set_xlabel("r")
ax.set_ylabel("H2")
fig.tight_layout()
plt.show()
"""