
from bisect import bisect
import numpy as np
import numpy.ma as ma
import pickle
import scipy
import time

from h2py import util


def compute_H2(
    vcf_file,
    bed_file=None,
    rec_map_file=None,
    mut_map_file=None,
    pop_file=None,
    region=None,
    r_bins=None,
    cM_bins=None,
    bp_bins=None,
    phased=False,
    min_reg_len=None,
    compute_denom=True,
    compute_two_sample=True,
    verbose=True
):
    """
    Compute H2 and H statistics from a region in a .vcf file.

    :param pop_file: form `sample_id` `pop_id`
    """
    if compute_denom is True and bed_file is None:
        raise ValueError('must provide bedfile to compute denominator')

    vcf_chrom, sample_ids, positions, genotypes = util.read_genotypes(
        vcf_file,
        bed_file=bed_file,
        min_reg_len=min_reg_len,
        region=region
    )

    if pop_file is not None:
        pop_ids, genotypes = subset_genotypes_by_pop(
            pop_file, sample_ids, genotypes
        )
    else:
        pop_ids = sample_ids

    if rec_map_file is not None:
        if r_bins is not None:
            bins = util.map_function(r_bins)
            ret_bins = r_bins
        elif cM_bins is not None:
            bins = cM_bins
            ret_bins = cM_bins
        else:
            raise ValueError('must provide r or cM bins')
        
        pos_map = util.read_recombination_map(rec_map_file, positions)

    else:
        if bp_bins is not None:
            bins = bp_bins
            ret_bins = bp_bins
        else:
            raise ValueError('must provide bp bins')
        
        pos_map = positions

    if region is not None:
        start, l_end, r_end = region
    else:
        start, l_end, r_end = None, None, None

    stats = {}

    num_H = get_H_statistics(
        genotypes, 
        compute_two_sample=compute_two_sample,
        verbose=verbose
    )
    num_H2 = get_H2_statistics(
        genotypes,
        pos_map,
        bins=bins,
        l_end=l_end,
        phased=phased,
        compute_two_sample=compute_two_sample,
        verbose=verbose
    )
    stats['nums'] = np.vstack((num_H2, num_H[np.newaxis]))

    if compute_denom:
        _, bed_positions = util.read_bedfile_positions(bed_file, region=region)

        denom_H = _denominator_H(bed_positions)

        if rec_map_file is not None:
            bed_map = util.read_recombination_map(rec_map_file, bed_positions)
        else:
            bed_map = bed_positions

        if mut_map_file is None:
            denom_H2 = _denominator_H2(
                bed_map, 
                bins=bins, 
                l_end=l_end, 
                mut_map=None,
                verbose=verbose
            )
            stats['denoms'] = np.append(denom_H2, denom_H)
            stats['means'] = np.divide(
                stats['nums'], stats['denoms'], where=stats['denoms'] > 0
            )
        
        else:
            mut_map = util.read_mutation_map(mut_map_file, bed_positions)
            mut_prod_sums, mut_stats = _denominator_H2(
                bed_map,
                bins=bins,
                l_end=l_end,
                mut_map=mut_map,
                verbose=verbose
            )
            stats['denoms'] = np.append(mut_prod_sums, denom_H)
            stats['mut_stats'] = mut_stats
            temp_denom = np.append(
                mut_prod_sums / mut_stats['mean_mut_prod'], denom_H
            )
            stats['means'] = np.divide(
                stats['nums'], 
                temp_denom[:, None], 
                where=(temp_denom > 0)[:, None]
            )

    stats['bins'] = ret_bins
    stats['pop_ids'] = pop_ids
    stats['num_sites'] = denom_H

    return stats


def subset_genotypes_by_pop(pop_file, sample_ids, genotypes):
    """
    Read a population file mapping sample ids to population ids with format
    name_in_vcf pop_id
    and return the subset of the genotype array which is specified therein.
    We work with individual samples, so no two individuals should share a 
    population id.
    """        
    idxs = []
    pop_ids = []

    with open(pop_file, 'r') as f:
        for l in f:
            sample, pop = l.split()
            if sample not in sample_ids:
                raise ValueError(f'sample {pop} is not present in sample_ids')
            if pop in pop_ids:
                raise ValueError(f'population names must be unique')
            idxs.append(sample_ids.index(sample))
            pop_ids.append(pop)

    genotypes = genotypes[idxs]

    return pop_ids, genotypes


def get_H_statistics(genotypes, compute_two_sample=True, verbose=False):
    """

    """
    num_pops = len(genotypes)
    if compute_two_sample: 
        num_stats = num_pops + util.n_choose_2(num_pops)
    else:
        num_stats = num_pops
    num_H = np.zeros(num_stats, dtype=np.float64)
    
    k = 0
    for i in range(len(genotypes)):
        for j in range(len(genotypes)):
            if i == j:
                num_H[k] = _one_sample_H(genotypes[i])
                k += 1
            elif j > i:
                if not compute_two_sample:
                    continue
                num_H[k] = _two_sample_H(genotypes[[i, j]])
                k += 1

    if verbose:
        print(util.get_time(), f'computed H for {genotypes.shape[1]} sites')

    return num_H


def _one_sample_H(genotypes):
    """
    """
    H = (genotypes[:, 0] != genotypes[:, 1]).sum()
    return H


def _two_sample_H(genotypes):
    """
    Takes an array of genotypes for two samples; shape (2, num_sites, 2)
    """
    H = (genotypes[0, :, :, None] != genotypes[1, :, None]).sum() / 4
    return H


def _denominator_H(bed_positions):
    """
    
    """
    return len(bed_positions)


def get_H2_statistics(
    genotypes, 
    pos_map, 
    bins=None,
    l_end=None,
    phased=False,
    compute_two_sample=True,
    verbose=False
):
    """
    """
    if phased:
        one_sample_func = _one_sample_haplotype_H2
        two_sample_func = _two_sample_haplotype_H2
    else:
        one_sample_func = _one_sample_genotype_H2
        two_sample_func = _two_sample_genotype_H2

    if l_end is not None:
        llim = np.searchsorted(pos_map, l_end)
    else:
        llim = None

    num_bins = len(bins) - 1
    num_pops = len(genotypes)
    if compute_two_sample: 
        num_stats = num_pops + util.n_choose_2(num_pops)
    else:
        num_stats = num_pops
    num_H2 = np.zeros((num_bins, num_stats), dtype=np.float64)

    k = 0
    for i in range(len(genotypes)):
        for j in range(len(genotypes)):
            if i == j:
                num_H2[:, k] = one_sample_func(
                    genotypes[i], pos_map, bins, llim=llim
                )
                k += 1
            elif j > i:
                if not compute_two_sample:
                    continue
                num_H2[:, k] = two_sample_func(
                    genotypes[[i, j]], pos_map, bins, llim=llim
                )
                k += 1
    if verbose:
        print(util.get_time(), f'computed H2 for {genotypes.shape[1]} sites')

    return num_H2


def _one_sample_genotype_H2(genotypes, pos_map, bins, llim=None): 
    """
    
    """
    het_indicator = genotypes[:, 0] != genotypes[:, 1]
    het_map = pos_map[het_indicator]
    H2 = count_num_pairs(het_map, bins=bins, llim=llim)
    return H2


def _two_sample_genotype_H2(genotypes, pos_map, bins, llim=None): 
    """
    
    """
    site_sampling_pr = \
        (genotypes[0, :, :, None] != genotypes[1, :, None]).sum((2, 1)) / 4
    H2 = compute_prod_sums(site_sampling_pr, pos_map, bins, llim=llim)
    return H2


def _one_sample_haplotype_H2():
    """
    """
    
    return


def _two_sample_haplotype_H2():
    """
    """
    
    return


def _denominator_H2(
    pos_map,
    bins=None,
    l_end=None,
    mut_map=None,
    verbose=None
):
    """
    Comoute the denominator for the H2 statistic.
    """    
    if l_end is not None:
        llim = np.searchsorted(pos_map, l_end)
    else:
        llim = None

    if mut_map is None:
        denom = count_num_pairs(pos_map, bins, llim=llim, verbose=None)
        ret = denom
        print(util.get_time(), f'computed denominator for {len(pos_map)} sites')

    else:
        mut_prod_sums = _mut_prod_sums(
            pos_map, 
            mut_map, 
            bins, 
            llim=llim,
            verbose=None
        )
        print(util.get_time(), f'computed denominator for {len(pos_map)} sites')
        mut_stats = {}


        mut_stats['num_sites'] = len(mut_map)
        mut_stats['mean_mut'] = np.mean(mut_map)
        mut_stats['sqr_mean_mut'] = np.mean(mut_map) ** 2
        mut_stats['mean_mut_prod'] = _mean_mut_prod(mut_map)
        whole_bin = np.array([bins[0], bins[-1]])
        mut_stats['num_pairs'] = count_num_pairs(pos_map, whole_bin, llim=llim)
        print(util.get_time(), f'computed mut stats for {len(pos_map)} sites')

        ret = (mut_prod_sums, mut_stats)

    return ret


def count_num_pairs(site_map, bins, llim=None, verbose=None):
    """
    Get counts of site pairs binned by their recombination distance. 
    """
    if not llim:
        llim = len(site_map)

    if not verbose or verbose is False:
        verbose = 1e10

    cum_nums = np.zeros(len(bins), dtype=int)

    for i, rl in enumerate(site_map[:llim]):
        edges = np.searchsorted(site_map[i + 1:], rl + bins)
        cum_nums += edges

        if i % verbose == 0 and i > 0:
            print(util.get_time(), f'num pairs counted at site {i}')

    _num_pairs = np.diff(cum_nums)
    num_pairs = ma.array(_num_pairs, mask=_num_pairs == 0)

    return num_pairs


def compute_prod_sums(site_vals, site_map, bins, llim=None, verbose=None):
    """
    Get sums of site-pair products binned by recombination distance.
    """
    if len(site_vals) != len(site_map):
        raise ValueError('rmap length mismatches umap')
    
    if not llim:
        llim = len(site_map)

    if not verbose:
        verbose = 1e10

    cum_vals = np.cumsum(site_vals)
    cum_prods = np.zeros(len(bins), dtype=float)

    for i, (rl, lval) in enumerate(zip(site_map[:llim], site_vals[:llim])):
        if lval > 0:
            edges = np.searchsorted(site_map[i + 1:], rl + bins)
            cum_prods += lval * cum_vals[i:][edges]

            if i % verbose == 0 and i > 0:
                print(util.get_time(), f'site prods computed at site {i}')

    _prod_sums = np.diff(cum_prods)
    prod_sums = ma.array(_prod_sums, mask=_prod_sums == 0)

    return prod_sums


def _mut_prod_sums(rec_map, mut_map, bins, llim=None, verbose=None):
    """
    """
    # compute sums of products of left * right locus mutation rates
    if len(mut_map) != len(rec_map):
        raise ValueError('recombination / mutation map lengths do not match')
    
    if llim is None:
        llim = len(rec_map)

    if verbose is None:
        verbose = 1e10

    cum_mut_map = np.cumsum(mut_map)
    cum_prods = np.zeros(len(bins), dtype=np.float64)

    for i, (rl, ul) in enumerate(zip(rec_map[:llim], mut_map[:llim])):
        if ul > 0:
            edges = np.searchsorted(rec_map[i + 1:], rl + bins)
            cum_prods += ul * cum_mut_map[i:][edges]

            if i % verbose == 0 and i > 0:
                print(util.get_time(), f'u prods computed at site {i}')

    _prod_sums = np.diff(cum_prods)
    prod_sums = ma.array(_prod_sums, mask=_prod_sums == 0)

    return prod_sums
 

def _mean_mut_prod(mut_map):
    """
    Get the mean of u * u site products across a map
    """
    cum_mut_map = np.cumsum(mut_map)
    site_cum_mut = cum_mut_map[-1] - cum_mut_map
    sum_prods = (site_cum_mut * mut_map).sum()

    len_mut_map = len(mut_map)
    num_prods = len_mut_map * (len_mut_map - 1) // 2

    mean_mut_prod = sum_prods / num_prods
    return mean_mut_prod


def _count_num_pairs(rmap, bins, llim=None):
    """
    given a recombination map, bin each pair of mapped sites by looping over
    the map. 
    
    this is a 'naive' function intended for testing against vectorized and 
    considerably faster pair-counting functions. 

    r_map and bins must have the same linear scale; typically this will be the
    centimorgan (cM). where it is useful to count site pairs whose left
    (lower-indexed) site is below some position, a left_bound may be applied;
    this is an index of r_map and counting ceases when it is reached.
    """
    if not llim:
        llim = len(rmap)
    num_bins = len(bins) - 1
    num_pairs = np.zeros(len(bins) - 1, dtype=int)
    for i, rl in enumerate(rmap[:llim]):
        for rr in rmap[i + 1:]:
            distance = rr - rl
            bin_idx = bisect(bins, distance) - 1
            if bin_idx < num_bins:
                if bin_idx >= 0:
                    num_pairs[bin_idx] += 1
    num_pairs = ma.array(num_pairs, mask=num_pairs == 0)
    return num_pairs


def _get_num_pairs_arr(rmap, bins, llim=None):
    """
    """
    if not llim:
        llim = len(rmap)
    num_bins = len(bins) - 1
    nums_arr = np.zeros((llim, len(bins) - 1), dtype=int)
    for i, rl in enumerate(rmap[:llim]):
        for rr in rmap[i + 1:]:
            distance = rr - rl
            bin_idx = bisect(bins, distance) - 1
            if bin_idx < num_bins:
                if bin_idx >= 0:
                    nums_arr[i, bin_idx] += 1
    return nums_arr


def _count_sums_prods(umap, rmap, bins, llim=None):
    """
    """
    if len(umap) != len(rmap):
        raise ValueError('umap and rmap have mismatched lengths')
    if not llim:
        llim = len(rmap) 
    num_bins = len(bins) - 1
    sums_prods = np.zeros(len(bins) - 1, dtype=float)
    for i, (rl, ul) in enumerate(zip(rmap[:llim], umap[:llim])):
        for rr, ur in zip(rmap[i + 1:], umap[i + 1:]):
            distance = rr - rl
            bin_idx = bisect(bins, distance) - 1
            if bin_idx < num_bins:
                if bin_idx >= 0:
                    sums_prods[bin_idx] += ul * ur
    sums_prods = ma.array(sums_prods, mask=sums_prods == 0)
    return sums_prods


def bootstrap_H2(
    regions, 
    num_reps=None, 
    num_samples=None,
    mut_weighted=False,
    to_mean_mut=None
):
    """
    Bootstrap H2 from genomic blocks. Takes a dictionary of bootstrap block 
    statistics as input. These statistics must be sums and not means.
    """
    keys = list(regions.keys())
    for key in keys:
        for field in ['pop_ids', 'bins']:
            if not np.all(regions[key][field] == regions[keys[0]][field]):
                raise ValueError(f'block {key} has mismatched pop_ids/bins')
            
    num_regions = len(regions)

    if num_reps is None:
        num_reps = num_regions

    if num_samples is None:
        num_samples = num_regions

    print(util.get_time(), f'bootstrapping with {num_reps} reps of '
          f'{num_samples} samples')

    if mut_weighted is False:
        stats = _bootstrap(regions, num_reps, num_samples)

    else:
        stats = _mut_weighted_bootstrap(
            regions,
            num_reps,
            num_samples,
            to_mean_mut=to_mean_mut
        )
    
    return stats


def _bootstrap(
    regions,
    num_reps,
    num_samples,
):
    """
    """
    region0 = regions[next(iter(regions))]
    num_bins, num_stats = region0['nums'].shape
    reps = np.zeros((num_reps, num_bins, num_stats), dtype=np.float64)

    for rep in range(num_reps):
        samples = np.random.choice(regions, num_samples, replace=True)
        rep_sums = np.zeros((num_bins, num_stats), dtype=np.float64)
        rep_denoms = np.zeros(num_bins, dtype=np.float64)
        for key in samples:
            rep_sums += regions[key]['nums']
            rep_denoms += regions[key]['denoms']
        reps[rep] = rep_sums / rep_denoms[:, np.newaxis]

    means = (
        np.array(regions[reg]['nums'] for reg in regions).sum(0) / 
        np.array(regions[reg]['denoms'] for reg in regions).sum(0)
    )
    varcovs = np.array(
        [np.cov(reps[:, b], rowvar=False) for b in range(num_bins)]
    )

    stats = {}
    stats['pop_ids'] = region0['pop_ids']
    stats['bins'] = region0['bins']
    stats['means'] = means
    stats['covs'] = varcovs

    return stats


def _mut_weighted_bootstrap(
    regions,
    num_reps,
    num_samples,
    to_mean_mut=None,
):
    """

    """
    region0 = regions[next(iter(regions))]
    num_bins, num_stats = region0['nums'].shape
    sums = np.zeros((num_bins, num_stats), dtype=np.float64)
    denom_sums = np.zeros(num_bins, dtype=np.float64)

    for key in regions:
        sums += regions[key]['nums']
        denom_sums += regions[key]['denoms']

    if to_mean_mut is None:
        sum_mut = 0
        sum_sites = 0
        for key in regions:
            num_sites = regions[key]['mut_stats']['num_sites']
            sum_mut += num_sites * regions[key]['mut_stats']['mean_mut']
            sum_sites += num_sites
        print(sum_mut, sum_sites)
        mean_mut = sum_mut / sum_sites
        print(util.get_time(), f'normalizing to genome-average u {mean_mut}')
        mut_fac = mean_mut ** 2 
    else:
        print(util.get_time(), f'normalizing to u = {to_mean_mut}')
        mut_fac = to_mean_mut ** 2
    
    denom_sums[:-1] /= mut_fac
    means = sums / denom_sums[:, np.newaxis]

    region0 = regions[next(iter(regions))]
    num_bins, num_stats = region0['nums'].shape

    reps = np.zeros((num_reps, num_bins, num_stats), dtype=np.float64)
    labels = list(regions.keys())

    for rep in range(num_reps):
        samples = np.random.choice(labels, num_samples, replace=True)
        rep_sums = np.zeros((num_bins, num_stats), dtype=np.float64)
        rep_denom_sums = np.zeros(num_bins, dtype=np.float64)
        for key in samples:
            rep_sums += regions[key]['nums']
            rep_denom_sums += regions[key]['denoms']
        
        if to_mean_mut:
            mut_fac = to_mean_mut ** 2
        else:
            sum_mut = 0
            sum_sites = 0
            for key in regions:
                num_sites = regions[key]['mut_stats']['num_sites']
                sum_mut += regions[key]['mut_stats']['mean_mut'] * num_sites
                sum_sites += num_sites
            mut_fac = (sum_mut / sum_sites) ** 2
                
        rep_denoms = rep_denom_sums / mut_fac     
        reps[rep] = rep_sums / rep_denoms[:, np.newaxis]
    
    varcovs = np.array(
        [np.cov(reps[:, b], rowvar=False) for b in range(num_bins)]
    )

    stats = {}
    stats['pop_ids'] = region0['pop_ids']
    stats['bins'] = region0['bins']
    stats['means'] = means
    stats['covs'] = varcovs

    return stats


def __bootstrap_H2(
    num_H2, 
    num_pairs, 
    sample_ids=None, 
    bins=None,
    num_bootstraps=None, 
    sample_size=None,
    return_distr=False
):
    """
    Bootstrap H2 from genomic blocks and return an H2stats instance holding
    bootstrap means and variances/covariances. Operates on masked arrays
    of counts.

    :param num_H2: 3dim, shape (num_windows, num_bins + 1, num_statistics)
        array of raw H2 counts with H at highest dim1 index.
    :param num_pairs: 2dim, shape (num_windows, num_bins + 1) 
        array of site pair counts or weighting factors, with site counts in
        the highest dim1 index.
    """
    assert num_H2.shape[:2] == num_pairs.shape
    num_windows, num_bins, num_stats = num_H2.shape
    sample_size = num_stats

    num_distr = np.zeros((num_bootstraps, num_bins, num_stats), dtype=float)
    denom_distr = np.zeros((num_bootstraps, num_bins), dtype=float)

    for i in range(num_bootstraps):
        sampled = np.random.Generator.integers(0, sample_size, num_stats)
        num_distr[i] = num_H2[sampled].sum(0)
        denom_distr[i] = num_pairs[sampled].sum(0)

    rep_H2 = num_distr / denom_distr[:, :, np.newaxis]
    if return_distr:
        ret = rep_H2
    else:
        covs = np.array(
            [np.cov(rep_H2[:, i], rowvar=False) for i in range(num_bins)]
        )
        means = num_distr.sum(0) / denom_distr.sum(0)
        ret = 0 
    return ret


def subset_H2_dict():

    return 
