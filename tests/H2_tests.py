"""
tests for the observed discrepancy between long-distance H2 and H^2
"""
import demes
import matplotlib
import matplotlib.pyplot as plt
import msprime
import numpy as np

from archaic import utils, one_locus, two_locus, masks, simulation, parsing
from archaic.spectra import H2Spectrum


# file names
chr1 = '/home/nick/Projects/archaic/tests/chr1_H2.npz'
chr22 = '/home/nick/Projects/archaic/tests/chr22_H2.npz'


# definitions
_sample_idx = 54  # should be one-sample, Yoruba-3
_bounds = 'bounds'
_n_sites = 'n_sites'
_n_site_pairs = 'n_site_pairs'
_n_H = 'H_counts'
_n_H2 = 'H2_counts'
_r_bins = 'r_bins'


"""
large-distance H2 is, in general, almost always about 0.95 to 0.9 of H**2.

only in archaic one-sample H2 is large-bin H2 greater than H2.
"""


def n_choose_2(n):
    #
    return int(n * (n - 1) * 0.5)


def chrom_test(fname):
    #
    file = np.load(fname)
    sites = file[_n_sites]
    pairs = file[_n_site_pairs]
    H_count = file[_n_H]
    H2_count = file[_n_H2]
    r = file[_r_bins][1:]

    bounds = file[_bounds]
    centromere = np.where(np.diff(bounds) > 0)[0][0] + 1

    # check H counts. pre-centromere
    H_sample = H_count[:, _sample_idx]
    H2_sample = H2_count[:, _sample_idx]
    assert n_choose_2(H_sample[:centromere].sum()) == H2_sample[:centromere].sum()
    # post-centromere
    assert n_choose_2(H_sample[centromere:].sum()) == H2_sample[centromere:].sum()
    # whole-chromosome
    assert n_choose_2(H_sample.sum()) > H2_sample.sum()
    print(f'{H2_sample.sum() / n_choose_2(H_sample.sum())} H excluded by bounds')

    # check site counts
    assert n_choose_2(sites[:centromere].sum()) == pairs[:centromere].sum()
    # post-centromere
    assert n_choose_2(sites[centromere:].sum()) == pairs[centromere:].sum()
    # whole-chromosome
    assert n_choose_2(sites.sum()) > pairs.sum()
    print(f'{pairs.sum() / n_choose_2(sites.sum())} sites excluded by bounds')


    return 0


def accro_chrom_test(fname):
    # for chromosomes without centromeres. different behavior
    file = np.load(fname)
    sites = file[_n_sites]
    pairs = file[_n_site_pairs]
    H_count = file[_n_H]
    H2_count = file[_n_H2]
    r = file[_r_bins][1:]

    # if the r-bins extended from 0 to 0.5, this should be True
    assert n_choose_2(sites.sum()) == pairs.sum()
    print(f'n_sites: {sites.sum()}')
    print(f'n_pairs: {pairs.sum()}')

    Hct = H_count[:, _sample_idx].sum()
    H2ct = H2_count[:, _sample_idx, :].sum(0)
    # total H pair-count should be H choose 2
    assert n_choose_2(Hct) == H2ct.sum()

    H = Hct / sites.sum()
    H2 = H2ct / pairs.sum(0)

    plt.plot(r, H2, marker='x', color='black', label='H2')
    plt.plot([r[0], r[-1]], [H ** 2] * 2, color='red', label='H')
    plt.xscale('log')
    plt.ylim(0, )
    plt.show()
    return 0






"""
approximating H2 from binned H?
"""


_vcf = '/home/nick/Projects/archaic/data/chroms/main/chr1.vcf.gz'
_rmap = '/home/nick/Projects/archaic/data/maps/omni/YRI/YRI-1-final.txt'
_mask = '/home/nick/Projects/archaic/data/masks/main/chr1_main.bed'




def approx_H2(windows):
    # windows is 1dim vec of edges
    mask = masks.Mask.from_bed_file(_mask)
    variants = one_locus.VariantFile(_vcf, mask=mask)
    # we look only at Yoruba-3
    yor3_idx = 9
    genos = variants.genotypes[:, yor3_idx, :]
    positions = mask.positions
    _positions = variants.positions
    nH = np.zeros(len(windows) - 1)
    nsites = np.zeros(len(windows) - 1)
    for i in range(len(windows) - 1):
        window = np.array([windows[i], windows[i + 1]])
        start, stop = np.searchsorted(positions, window)
        n_sites = stop - start
        _start, _stop = np.searchsorted(_positions, window)
        _genos = genos[_start:_stop]
        H_count = np.count_nonzero(_genos[:, 0] != _genos[:, 1])
        if n_sites > 0:
            nH[i] = H_count
            nsites[i] = n_sites
        else:
            if H_count != 0:
                print('H counted in area of 0 coverage')
    mids = windows[:-1] + np.diff(windows) / 2
    r_map = two_locus.get_r_map(_rmap, mids)
    d_bins = two_locus.map_function(r_bins)

    nH2 = np.zeros(len(d_bins) - 1)
    npairs = np.zeros(len(d_bins) - 1)
    for i in range(len(windows) - 1):
        focal_d = r_map[i]
        bin_idxs = np.digitize(r_map[i + 1:] - focal_d, d_bins) - 1
        for k in range(len(d_bins) - 1):
            nH2[k] += (nH[i] * nH[i + 1:][bin_idxs == k]).sum()
            npairs[k] += (nsites[i] * nsites[i + 1:][bin_idxs == k]).sum()
    return nH2 / npairs


def plot_ratio():
    labels = []
    last = []
    secondlast = []
    thirdlast = []
    for i, (x, y) in enumerate(ids):
        if x == y:
            _H = H[i]
            _H2 = H2[i, -1]
            last.append( H2[i, -1] / H[i] ** 2)
            secondlast.append(H2[i, -2] / H[i] ** 2 )
            thirdlast.append( H2[i, -3] / H[i] ** 2)
            labels.append(x)
    x = np.arange(len(labels))
    plt.scatter(x, last, color='black', marker='x', label='$r\in[0.32, 0.50)$')
    plt.scatter(x, secondlast, color='red', marker='x',
                label='$r\in[0.18, 0.32)$')
    plt.scatter(x, thirdlast, color='blue', marker='x',
                label='$r\in[0.10, 0.18)$')
    plt.xticks(x, labels, rotation=90)
    plt.ylabel(r'$H_2/H^2$')
    plt.grid(alpha=0.2)
    plt.ylim(0.9, 1.1)
    plt.legend()
    plt.savefig(
        '/home/nick/Projects/archaic/statistics/corrected_omniYRI/Hsquared/largestbin',
        bbox_inches='tight', dpi=300)
    return 0


r_bins = np.array([0.00000000e+00, 1.00000000e-06, 1.77827941e-06, 3.16227766e-06,
       5.62341325e-06, 1.00000000e-05, 1.77827941e-05, 3.16227766e-05,
       5.62341325e-05, 1.00000000e-04, 1.77827941e-04, 3.16227766e-04,
       5.62341325e-04, 1.00000000e-03, 1.77827941e-03, 3.16227766e-03,
       5.62341325e-03, 1.00000000e-02, 1.77827941e-02, 3.16227766e-02,
       5.62341325e-02, 1.00000000e-01, 1.77827941e-01, 3.16227766e-01,
       5.00000000e-01])


def simulate_variant_u():

    # are peculiar patterns in the distribution of H2/H^2 related to patterns of
    # H along the chromosome?

    L = 1e8
    vcf_fname = 'u_test.vcf'

    def write_mask_file(L):
        #
        regions = np.array([[0, L]], dtype=np.int64)
        mask_fname = f'mask{int(L / 1e6)}Mb.bed'
        chrom_num = 'chr0'
        masks.write_regions(regions, mask_fname, chrom_num)
        return mask_fname

    def write_map_file(L, r):
        #
        cM = two_locus.map_function(r) * L
        map_fname = f'map{int(L / 1e6)}Mb.txt'
        with open(map_fname, 'w') as file:
            file.write('Position(bp)\tRate(cM/Mb)\tMap(cM)\n')
            file.write('1\t0\t0\n')
            file.write(f'{int(L)}\t0\t{cM}')
        return map_fname

    # get a rate map
    x = np.arange(L + 1)
    # mutation rate is 1e-8 at center, 1.5e-8 at ends
    rate = (1 + 2 / (L ** 2) * (x[:-1] - L / 2) ** 2) * 1e-8
    rate_map = msprime.RateMap(position=x, rate=rate)
    b = demes.Builder(generation_time=25, time_units="years")
    b.add_deme("X", epochs=[dict(start_size=10000, end_time=0)])
    g = b.resolve()
    simulation.simulate(g, L, u=rate_map, out_fname=vcf_fname)


    mask_fname = write_mask_file(L)
    map_fname = write_map_file(L, 1e-8)

    size = 1e7
    windows = np.array([[i * size, (i+1) * size] for i in range(int(L / size))])
    bounds = np.full((len(windows)), L)
    H2_dict = parsing.parse_H2(
            mask_fname,
            vcf_fname,
            map_fname,
            windows=windows,
            bounds=bounds,
            r_bins=r_bins
    )
    H2_stats = parsing.bootstrap_H2([H2_dict], n_iters=100)
    return H2_stats


















