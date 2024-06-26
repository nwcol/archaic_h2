#!/bin/bash

### elements needed in pwd: ###

# strict 1000 genomes masks
# exon masks [define better]
# directories containing 4 archaic genomes
# tarball containing 279 SGDP genomes
# centromeres.txt, tab-seperated file defining positions of chromosome centromeres
# samples.txt, space-seperated file defining samples to extract from SGDP tarball


loc=$pwd
temp=./temp

mkdir $temp
mkdir $temp/masks


for name in Altai Chagyrskaya Denisova Vindija
do
	for i in {1..22}
	do
		python -m archaic.process.mask_from_vcf -v ArchaicGenomes/${name}/*chr${i}_*.vcf.gz -o $temp/masks/chr${i}_${name}.bed -n $i
	done
done













_names=$(cut samples.txt -f 1 -d " ")
names=$(cut samples.txt -f 2 -d " ")

for _name in "${_names[@]}"
do
	echo $_name
	# pull genome out of the simons tarball
done

# construct masks for archaic genomes
# intersect masks with each other
# intersect with strict mask and print overlap [???]
# process out exons, delete regions <50 bp in length
# merge .vcfs, treating missing sites as ref
# rename samples
# split non-acrocentric chromosomes into A and B components


