"""
for a given chromosome, conduct 4 simulations with empirical/uniform r/u maps,
then parse each simulated chromosome with combinations of empirical/uniform maps
"""
import demes
import numpy as np
import sys

from archaic import util, parsing, simulation


cluster = sys.argv[1]
process = sys.argv[2]
tag = f'{cluster}-{process}'
n = process


# CONSTANTS
warr = np.loadtxt(f'blocks_{n}.txt')
windows = warr[:, :2]
bounds = warr[:, 2]

graph_fname = 'g.yaml'
rmap_fname = f'YRI-{n}-final.txt'
umap_fname = f'u_{n}_10kb.bedgraph.gz'
mask_fname = f'roulette_isec_{n}.bed.gz'
bins = np.loadtxt('fine_bins.txt')


if __name__ == '__main__':
    graph = demes.load(graph_fname)

    # measure mean r, u in the empirical maps
    regions = utils.read_mask_file(mask_fname)
    positions = utils.get_mask_positions(regions)

    edges, windowed_u = utils.read_u_bedgraph(umap_fname)
    idx = np.searchsorted(edges[1:], positions)
    mean_u = windowed_u[idx].mean()
    print(utils.get_time(), 'mean u in mask: {mean_u}')

    L = positions[-1]
    map_span = utils.read_map_file(rmap_fname, [0, L])
    mean_r_cM = (map_span[1] - map_span[0]) / L
    mean_r = mean_r_cM / 100
    print(util.get_time(), 'mean r across map: {mean_r}')
    # create a uniform r-map file
    unif_rmap_fname = f'unif_rmap_{n}.txt'
    with open(unif_rmap_fname, 'w') as file:
        file.write(
            'Position(bp)\tMap(cM)\n'
            f'0\t{map_span[0]}\n'
            f'{L}\t{map_span[1]}'
        )

    # simulate chromosomes
    vcf_fnames = []
    for rname, r in zip(['unif', 'emp'], [mean_r, rmap_fname]):
        for uname, u in zip(['unif', 'emp'], [mean_u, umap_fname]):
            vcf_fname = f'{rname}r_{uname}u_{tag}.vcf'
            simulation.simulate_chromosome(
                graph,
                vcf_fname,
                u=u,
                r=r,
                contig_id=1,
                L=L
            )
            vcf_fnames.append(vcf_fname)
            print(util.get_time(), f'saved simulation at {vcf_fname}')

    # parse statistics from simulated chromosomes
    for vcf_fname in vcf_fnames:
        base_name = vcf_fname.replace('.vcf', '').replace(f'_{tag}', '')
        for rname, rfile in zip(['unif', 'emp'], [unif_rmap_fname, rmap_fname]):
            # parsing with uniform u is the same as using the normal unscaled
            # function to compute H2
            weighted_u_stats = parsing.parse_weighted_H2(
                mask_fname,
                vcf_fname,
                rfile,
                umap_fname,
                bins,
                windows=windows,
                bounds=bounds
            )
            weighted_fname = f'{base_name}_{rname}r_weighted.npz'
            np.savez(weighted_fname, **weighted_u_stats)
            print(
                util.get_time(),
                f'weighted u stats with rmap {rname} @ {weighted_fname}'
            )

            unif_u_stats = parsing.parse_H2(
                mask_fname,
                vcf_fname,
                rfile,
                windows=windows,
                bounds=bounds,
                bins=bins,
            )
            unif_fname = f'{base_name}_{rname}r_normal.npz'
            np.savez(unif_fname, **unif_u_stats)
            print(
                util.get_time(),
                f'unif u stats with rmap {rname} @ {unif_fname}'
            )
