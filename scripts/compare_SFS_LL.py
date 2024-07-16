
import demes
import moments
import numpy as np


u = 1.35e-8


H2_graph_fname = '/home/nick/Projects/archaic/models/best_fit/ND_best_fit.yaml'
SFS_graph_fname = '/home/nick/Projects/archaic/models/with_SFS/top/ND_SFSfit_1750352_8_iter1.yaml'
data_fname = '/home/nick/Projects/archaic/models/SFS.npz'


SFS_file = np.load(data_fname)
_samples = SFS_file['samples']
_data = moments.Spectrum(SFS_file['SFS'], pop_ids=_samples)
deme_names = [d.name for d in demes.load(SFS_graph_fname).demes]
samples = list()
marginalize = list()
for i, _sample in enumerate(_samples):
    if _sample in deme_names:
        samples.append(_sample)
    else:
        marginalize.append(i)
data = _data.marginalize(marginalize)
L = SFS_file['n_sites']


config = {sample: 2 for sample in samples}


H2_graph = demes.load(H2_graph_fname)
data_for_H2 = moments.Demes.SFS(H2_graph, samples=config, u=u) * L


SFS_graph = demes.load(SFS_graph_fname)
data_for_SFS = moments.Demes.SFS(SFS_graph, samples=config, u=u) * L


lik_H2_graph = moments.Inference.ll(data_for_H2, data)
lik_SFS_graph = moments.Inference.ll(data_for_SFS, data)


print(f'LL of graph inferred with H2: {np.round(lik_H2_graph, 2)}')
print(f'LL of graph inferred with SFS: {np.round(lik_SFS_graph, 2)}')

