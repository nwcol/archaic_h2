import numpy as np

import matplotlib.pyplot as plt

import matplotlib


if __name__ == "__main__":
    plt.rcParams['figure.dpi'] = 100
    matplotlib.use('Qt5Agg')


def normal_lik(X, mu, sigma_2):
    marginals = ((1 / np.sqrt(2 * np.pi * sigma_2) *
                 np.exp(-0.5 * np.square(X - mu)/sigma_2)))
    lik = np.prod(marginals)
    return lik


def lik_ratio(x, mu_0, mu_1, sigma_2):
    lik_0 = ((1 / np.sqrt(2 * np.pi * 1) *
                 np.exp(-0.5 * np.square(x - mu_0)/1)))
    lik_1 = ((1 / np.sqrt(2 * np.pi * 1) *
                 np.exp(-0.5 * np.square(x - mu_1)/1)))
    return lik_0 / lik_1



