
from bisect import bisect
import copy
import demes
import numpy as np
import numpy.ma as ma
import pickle
import scipy
import time
import warnings

from h2py import util


def compute_H2(
    vcf_file,
    bed_file=None,
    rec_map_file=None,
    r=None,
    mut_map_file=None,
    pop_file=None,
    region=None,
    r_bins=None,
    cM_bins=None,
    bp_bins=None,
    phased=False,
    min_reg_len=None,
    compute_denom=True,
    compute_snp_denom=False,
    compute_two_sample=True,
    verbose=True
):
    """
    Compute H2 and H statistics from a region in a .vcf file.

    :parma r: constant recombination rate
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

    elif r is not None:
        pos_map = constant_rec_map(r, positions)

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
        elif r is not None:
            bed_map = constant_rec_map(r, bed_positions)
        else:
            bed_map = bed_positions

        # compute pair-count denominator
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
                stats['nums'], 
                stats['denoms'][:, np.newaxis], 
                where=(stats['denoms'] > 0)[:, np.newaxis]
            )
        
        # compute mutation-map-weighted denominator
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
                temp_denom[:, np.newaxis], 
                where=(temp_denom > 0)[:, np.newaxis]
            )
    
    # compute denominator from vcf sites only
    elif compute_snp_denom:
        denom_H = _denominator_H(positions)
        denom_H2 = _denominator_H2(
            pos_map,
            bins=bins,
            l_end=l_end,
            mut_map=None,
            verbose=verbose
        )
        stats['denoms'] = np.append(denom_H2, denom_H)
        stats['means'] = np.divide(
            stats['nums'], 
            stats['denoms'][:, np.newaxis], 
            where=(stats['denoms'] > 0)[:, np.newaxis]
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


def constant_rec_map(r, positions):
    """
    Obtain a recombination map for `positions` assuming a constant recombination
    rate `r`. Returns a map in units of cM.
    """
    cM_per_bp = util.map_function(r)
    rec_map = positions * cM_per_bp
    return


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
    num_diff_pairs = (genotypes[0, :, :, None] != genotypes[1, :, None]).sum()
    H = num_diff_pairs / 4
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
    het_sites = genotypes[:, 0] != genotypes[:, 1]
    het_map = pos_map[het_sites]
    H2 = count_num_pairs(het_map, bins=bins, llim=llim)
    return H2


def _two_sample_genotype_H2(genotypes, pos_map, bins, llim=None): 
    """
    
    """
    num_diff_pairs = genotypes[0, :, :, None] != genotypes[1, :, None]
    diff_probs = num_diff_pairs.sum((2, 1)) / 4
    H2 = compute_prod_sums(diff_probs, pos_map, bins, llim=llim)
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
    Compute the denominator for the H2 statistic.
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


"""
Bootstrapping
"""


def compute_regions_mean(regions, mut_weighted=False):
    """
    Compute the mean statistics across a dictionary of region statistics.
    """
    num_regions = len(regions)
    num_bins, num_stats = regions[next(iter(regions))]['nums'].shape

    if mut_weighted:
        sum_mut = 0
        sum_sites = 0
        for key in regions:
            num_sites = regions[key]['mut_stats']['num_sites']
            sum_mut += num_sites * regions[key]['mut_stats']['mean_mut']
            sum_sites += num_sites
        mean_mut = sum_mut / sum_sites
        mut_fac = mean_mut ** 2 

    sums = np.zeros((num_bins, num_stats), dtype=np.float64)
    denoms = np.zeros(num_bins, dtype=np.float64)

    for key in regions:
        sums += regions[key]['nums']

        if mut_weighted:
            denoms += regions[key]['denoms'] / mut_fac
        else:
            denoms += regions[key]['denoms']

    means = sums / denoms[:, np.newaxis]
    return means


def compute_varcov(reps):
    """
    Compute the variance-covariance matrix for each bin (and for H) from a 
    list or array of bootstrap replicates.
    """
    reps = np.asanyarray(reps)
    num_reps, num_bins, num_stats = reps.shape
    varcovs = np.zeros((num_bins, num_stats, num_stats), dtype=np.float64)

    for i in range(num_bins):
        varcovs[i] = np.cov(reps[:, i], rowvar=False)

    return varcovs


def get_bootstrap_reps(
    regions, 
    num_reps=None, 
    num_samples=None,
    mut_weighted=True
):
    """
    
    """
    num_regions = len(regions)

    if num_reps is None:
        num_reps = num_regions
    if num_samples is None:
        num_samples = num_regions

    labels = list(regions.keys())
    bootstrap_reps = []

    for rep in range(num_reps):
        samples = np.random.choice(labels, num_samples, replace=True)
        _regions = {sample: regions[sample] for sample in samples}
        bootstrap_reps.append(
            compute_regions_mean(_regions, mut_weighted=mut_weighted)
        )

    ex = regions[next(iter(regions))]
    data = {
        'pop_ids': ex['pop_ids'],
        'bins': ex['bins'],
        'means': compute_regions_mean(regions, mut_weighted=mut_weighted),
        'covs': compute_varcov(bootstrap_reps),
    }

    return data, bootstrap_reps


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

    if not mut_weighted:
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
    labels = list(regions.keys())

    for rep in range(num_reps):
        samples = np.random.choice(labels, num_samples, replace=True)
        rep_sums = np.zeros((num_bins, num_stats), dtype=np.float64)
        rep_denoms = np.zeros(num_bins, dtype=np.float64)
        for key in samples:
            rep_sums += regions[key]['nums']
            rep_denoms += regions[key]['denoms']
        reps[rep] = rep_sums / rep_denoms[:, np.newaxis]

    means = compute_regions_mean(regions)
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
                
        rep_denoms = np.zeros(num_bins, dtype=np.float64)
        # we don't want to scale H by the mutation rate
        rep_denoms[:-1] = rep_denom_sums[:-1] / mut_fac   
        rep_denoms[-1] = rep_denom_sums[-1]  
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


def replace_denominator(data, denom_data):
    """
    
    """
    ret = copy.deepcopy(data)

    ret['denoms'] = denom_data['denoms']

    if 'mut_stats' in denom_data:
        ret['mut_stats'] = denom_data['mut_stats']

    return ret


def subset_H2(
    data, 
    graph=None, 
    to_pops=None, 
    min_dist=None, 
    max_dist=None
):
    """
    Subset a dictionary of statistics by pop_id. If a graph is provided, subsets
    to the set of names which occur in both the data set and the graph.
    """
    if graph is not None:
        if to_pops is not None:
            warnings.warn('argument `to_pops` overriden by `graph`')
        if isinstance(graph, str):
            graph = demes.load(graph)
        graph_demes = [d.name for d in graph.demes]
        pop_ids = data['pop_ids']
        to_pops = [d for d in pop_ids if d in graph_demes]

    if to_pops is not None:
        pop_ids = data['pop_ids']
        labels = enumerate_labels(pop_ids)
        to_labels = enumerate_labels(to_pops)
        keep = np.array([labels.index(label) for label in to_labels])

        for key in ['nums', 'sums', 'means']:
            if key in data:
                data[key] = data[key][:, keep]
        
        if 'covs' in data:
            covs = data['covs']
            data['covs'] = np.stack([cov[np.ix_(keep, keep)] for cov in covs])
        
        data['pop_ids'] = to_pops

    if min_dist is not None or max_dist is not None:
        bins = data['bins']
        if min_dist is None:
            min_dist = 0
        if max_dist is None:
            max_dist = np.inf
        min_bin = np.searchsorted(bins, min_dist)
        max_bin = np.searchsorted(bins, max_dist)
        
        data['bins'] = bins[min_bin:max_bin + 1]

        for key in ['nums', 'sums', 'means']:
            if key in data:
                data[key] = data[key][min_bin:max_bin + 1]
        
        if 'covs' in data:
            data['covs'] = data['covs'][min_bin:max_bin + 1]

    return data


def enumerate_labels(pop_ids, has_two_sample=True):
    """
    Enumerate all the pairs of population ids, including self-pairs.
    """
    if has_two_sample:
        num_pops = len(pop_ids)
        labels = []
        for i in range(num_pops):
            for j in range(i, num_pops):
                labels.append((pop_ids[i], pop_ids[j]))
    else:
        labels = pop_ids
    return labels


# constants, defaults
_default_bins = np.logspace(-6, -1, 21)


