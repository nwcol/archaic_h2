"""
the idea here is to use a predicted B-map for a modern human population to
predict H2 using inferred local mutation rates and B-values

H2 is estimated as

H2[bin] = 16 * u^2[bin] * B^2[bin] * Ne^2 + 16 u^2[bin] * Cov(Tl, Tr)[bin]

"""
import argparse
import numpy as np

from archaic import util, counting, parsing


def get_args():
    # get args
    parser = argparse.ArgumentParser()
    parser.add_argument("-b", "--mask_fname", required=True)
    parser.add_argument("-r", "--rmap_fname", required=True)
    parser.add_argument("-o", "--umap_fname", required=True)
    parser.add_argument("-w", "--windows", default=None)
    parser.add_argument('--bins', default=None)
    return parser.parse_args()


mask_fname = '/home/nick/Projects/archaic/data/masks/exons_10kb/roulette-exons_10kb_2.bed.gz'
bmap_fname = '/home/nick/Projects/bgs_lmr/human_data/statistics/predictions/B-YRI-2-1kb.bedgraph'
umap_fname = '/home/nick/Data/roulette/umaps/umap_2.npy'
rmap_fname = '/home/nick/Data/rmaps/omni/YRI/YRI-2-final.txt.gz'
windows = np.loadtxt('/home/nick/Projects/archaic/data/windows/blocks/blocks_2.txt')
bins = np.loadtxt('/home/nick/Projects/archaic/data/misc/fine-bins.txt')
Ne = 24000


out_fname = '/home/nick/Projects/archaic/simulations/B_predictions/test-YRI-2.npz'


def compute_cov_T(bins, Ne):
    # compute the covariance of tree height under the coalescent with
    # recombination, for midpoints of r bins, in units of generations
    # from McVean 2002
    midpoints = bins[:-1] + np.diff(bins) / 2
    rho = 4 * Ne * midpoints
    cov = Ne ** 2 * (18 + rho) / (18 + 13 * rho + rho ** 2)
    return cov


if __name__ == '__main__':
    #
    regions = util.read_mask_file(mask_fname)
    positions = util.get_mask_positions(regions)

    r_map = util.read_map_file(rmap_fname, positions)

    u_map = np.load(umap_fname)[positions - 1]

    b_windows, data = util.read_bedgraph(bmap_fname)
    Bs = data['B']
    idx = np.searchsorted(b_windows[:, 1], positions)
    B_map = Bs[idx]

    print(util.get_time(), 'loaded data')

    u_prods = np.zeros((len(windows), len(bins) - 1))
    B_prods = np.zeros((len(windows), len(bins) - 1))
    num_pairs = np.zeros((len(windows), len(bins) - 1))

    for w, (wstart, lbound, rbound) in enumerate(windows):
        start = np.searchsorted(positions, wstart)
        r_end = np.searchsorted(positions, rbound)
        l_end = np.searchsorted(positions[start:], lbound)

        u_prods[w] = counting.count_weighted_site_pairs(
            u_map[start:r_end],
            r_map[start:r_end],
            bins,
            left_bound=l_end
        )
        B_prods[w] = counting.count_weighted_site_pairs(
            B_map[start:r_end],
            r_map[start:r_end],
            bins,
            left_bound=l_end
        )

        # denominator
        num_pairs[w] = counting.count_site_pairs(
            r_map[start:r_end],
            bins,
            left_bound=l_end
        )
        print(
            util.get_time(),
            f'computed pair counts for window {w} {wstart, lbound, rbound}'
        )

    print(util.get_time(), 'computed pair counts')

    mean_uprods = u_prods.sum(0) / num_pairs.sum(0)
    mean_Bprods = B_prods.sum(0) / num_pairs.sum(0)
    cov = compute_cov_T(bins, Ne)
    exp_H2 = 4 * mean_uprods * (4 * mean_Bprods * Ne ** 2 + cov)

    data = dict(
        Ne=Ne,
        num_pairs=num_pairs,
        Bprods=B_prods,
        uprods=u_prods,
        mean_uprods=mean_uprods,
        mean_Bprods=mean_Bprods,
        cov=cov,
        H2=exp_H2,
        ids=np.array([['null'], ['null']]),
        r_bin=bins
    )
    np.savez(out_fname, **data)
    print('saved data to file')


