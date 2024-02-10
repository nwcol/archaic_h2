

from util import sample_sets
from util import two_locus


ex = sample_sets.UnphasedSampleSet.read_chr(22)
two_locus.count_two_sample_het_pairs(ex, "Yoruba-1", "Khomani_San-2",
                                     two_locus.r_edges)
