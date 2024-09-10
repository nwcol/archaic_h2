
import demes
import numpy as np
import sys

from archaic import utils, parsing, simulation

# constants

process = sys.argv[1]

L = 250000000
r = 1e-8

warr = np.loadtxt('blocks_1.txt')
windows = warr[:, :2]
bounds = warr[:, 2]

graph_fname = 'g.yaml'
rmap_fname = 'uniform_rmap.txt'
umap_fname = 'empirical_umap_1.bedgraph.gz'
mask_fname = 'roulette_isec_1.bed.gz'
bins = np.loadtxt('fine_bins.txt')


if __name__ == '__main__':
    graph = demes.load(graph_fname)

    # measure mean r, u in the empirical maps
    regions = utils.read_mask_file(mask_fname)
    positions = utils.get_mask_positions(regions)

    edges, windowed_u = utils.read_u_bedgraph(umap_fname)
    idx = np.searchsorted(edges[1:], positions)
    mean_u = windowed_u[idx].mean()
    print(f'mean u in mask: {mean_u}')

    # simulate with empirical u-map and parse the simulated data
    emp_vcf_fname = f'empirical_u_{process}.vcf'
    simulation.simulate_chromosome(
        graph,
        emp_vcf_fname,
        u=umap_fname,
        r=r,
        contig_id=1,
        L=L
    )
    dic1 = parsing.parse_weighted_H2(
        mask_fname,
        emp_vcf_fname,
        rmap_fname,
        umap_fname,
        bins,
        windows=windows,
        bounds=bounds
    )
    np.savez(f'empirical_u_{process}_weighted.npz', **dic1)
    dic2 = parsing.parse_H2(
        mask_fname,
        emp_vcf_fname,
        rmap_fname,
        windows=windows,
        bounds=bounds,
        bins=bins,
    )
    np.savez(f'empirical_u_{process}_unif.npz', **dic2)
    print(f'empirical-u simulation complete')

    # simulate with uniform u-map
    unif_vcf_fname = f'uniform_u_{process}.vcf'
    simulation.simulate_chromosome(
        graph,
        unif_vcf_fname,
        u=mean_u,
        r=r,
        contig_id=1,
        L=L
    )
    dic1 = parsing.parse_weighted_H2(
        mask_fname,
        unif_vcf_fname,
        rmap_fname,
        umap_fname,
        bins,
        windows=windows,
        bounds=bounds
    )
    np.savez(f'uniform_u_{process}_weighted.npz', **dic1)
    dic2 = parsing.parse_H2(
        mask_fname,
        unif_vcf_fname,
        rmap_fname,
        windows=windows,
        bounds=bounds,
        bins=bins,
    )
    np.savez(f'uniform_u_{process}_unif.npz', **dic2)
    print(f'uniform-u simulation {process} complete')
