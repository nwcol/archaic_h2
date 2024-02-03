import numpy as np

import sys

from util import sample_sets

from util import two_locus


pair_count_path = sys.argv[1]
bed_path = sys.argv[2]
map_path = sys.argv[3]
output_path = sys.argv[4]
vcf_paths = sys.argv[5:]
pair_counts = np.loadtxt(pair_count_path)


def main(pair_counts, bed_path, map_path, output_path, vcf_paths):
    """
    """
    two_locus_hets = {}         
    for i, vcf_path in enumerate(vcf_paths):
        sample_set = vcf_samples.UnphasedSamples.one_file(vcf_path, bed_path,
            map_path)
        sample_ids = sample_set.sample_ids
        for sample_id in sample_ids:
            het_pair_counts = two_locus.count_heterozygous_pairs(sample_set, 
                sample_id)
            het = het_pair_counts / pair_counts
            if sample_id in two_locus_hets:
                two_locus_hets[sample_id].append(het)
            else: 
                two_locus_hets[sample_id] = [het]
        print(f"het for .vcf {i + 1} computed")

    for sample_id in two_locus_hets:
        filename = f"{output_path}{sample_id}_two_locus_het.txt"
        file = open(filename, 'w')
        np.savetxt(file, np.array(two_locus_hets[sample_id]))
        file.close()
        
    return 0


main(pair_counts, bed_path, map_path, output_path, vcf_paths)

