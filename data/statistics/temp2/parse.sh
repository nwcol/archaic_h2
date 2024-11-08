#!/bin/bash

for i in {5..19}; do
	parse_H2 -v ~/Projects/old_archaic/data/genotypes/main/chr${i}.vcf.gz \
		-b /home/nick/Projects/old_archaic/data/masks/exons_10kb/roulette-exons_10kb_${i}.bed.gz \
	        -bins bins.txt \
		-r /home/nick/Data/archaic/rmaps/omni/YRI/YRI-${i}-final.txt.gz \
	       	-o H2_chr${i}.pkl \
		-m /home/nick/Projects/archaic_h2/data/mutation_maps/mutation_map_hg19_chr${i}.npy \
		-R arms/arms_${i}.txt \
		--chrom chr${i} 
done

