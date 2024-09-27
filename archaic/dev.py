import numpy as np
from collections import namedtuple

import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib.ticker as mticker
from matplotlib.lines import Line2D

from archaic import util


_bins = np.array([0, 1e-6, 1e-5, 1e-4, 1e-3])


mapping = namedtuple('mapping', ['coords', 'vals'])


def make_rmap(res=10, L=1e+3, r=1e-6):
    #
    coords = np.arange(0, L + res, res, dtype=int)
    vals = np.cumsum(
        np.concatenate(([0], np.random.uniform(0, 2 * r * res, len(coords) - 1)))
    )
    return mapping(coords, vals)


rmap = make_rmap()
positions = np.arange(1001)
site_r = np.interp(
    positions,
    rmap.coords,
    rmap.vals
)


def _count_site_pairs(
    positions,
    rcoords,
    rmap,
    bins,
    left_bound=None
):
    # uses interpolation rather than searchsorting. currently WIP and not a
    # high priority
    if len(rcoords) != len(rmap):
        raise ValueError('rcoords length mismatches rmap')
    if left_bound is None:
        left_bound = len(positions)
    # interpolate r-map values
    site_r = np.interp(
        positions,
        rcoords,
        rmap,
        left=rmap[0],
        right=rmap[-1]
    )
    coords_in_pos_idx = np.searchsorted(positions, rcoords)
    coords_in_pos_idx[coords_in_pos_idx == len(positions)] -= 1
    edges = np.zeros(len(bins), dtype=int)

    for i, b in enumerate(bins):
        edges[i] = np.floor(
            np.interp(
                site_r[:left_bound] + b,
                rmap,
                coords_in_pos_idx,
                left=0,
                right=len(positions) - 1
            )
        ).sum()

    counts = np.diff(edges)

    return counts






def _integrate(rmap, objmap, bins):
    #
    obj = np.cumsum(rmap.coords)

    edges0 = np.interp(
        rmap.vals,
        rmap.vals + bins[0],
        obj,
        right=obj[-1] + 1,
    )
    edges1 = np.interp(
        rmap.vals,
        rmap.vals + bins[1],
        obj,
        right=obj[-1] + 1,
    )
    ret = np.ceil(edges1) - np.ceil(edges0)
    return


def __integrate(rmap, obj_mapping, bins):
    #
    # modify coords using obj_mapping values
    obj = rmap.coords

    _edges = np.interp(
        rmap.vals[:-1, np.newaxis] + bins[np.newaxis],
        rmap.vals,
        obj,
        right=obj[-1] + 1,
    )
    edges = np.ceil(_edges)

    # this is the critical part
    fac = np.diff(rmap.coords) \
          - np.searchsorted(rmap.vals, rmap.vals[:, None] + bins[None])
    ret = np.diff(fac * edges, axis=1).sum(0)

    #counts = (gaps[:, None] * np.diff(edges[:-1], axis=1)).sum(0)

    return ret


def plot(positions, site_r, bins, stop=10):
    #
    colors = list(cm.gnuplot(np.linspace(0, 0.95, len(bins) - 1)))

    fig, ax = plt.subplots(figsize=(8, 6), layout='constrained')

    for i in range(stop):
        pos = positions[i]
        idxs = np.searchsorted(site_r[i + 1:], site_r[i] + bins)

        for k in range(len(bins) - 1):
            y = positions[idxs[[k, k + 1]]] / 2
            x = pos + y
            ax.plot(x, y, color=colors[k], marker='D', markersize=4)
