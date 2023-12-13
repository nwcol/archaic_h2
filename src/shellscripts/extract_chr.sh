#!/bin/bash

# Parse one chromosome out of a .vcf.gz using bcftools, simplify it,
# and apply a mask

# arg 1: contig/chromosome to extract
# additional args: .vcf.gz to extract from


source /c/anaconda3/etc/profile.d/conda.sh
conda activate archaic_conda_env


i=0
for arg in "$@";
do
	if [ "$i" -ge 1 ]
	then
		if ! test -f "$arg.csi";
		then
			bcftools index "$arg"
		fi
		
		stem="chr$1_${arg:0:-7}" # .vcf.gz is 7 characters
		
		bcftools view -r "$1" -o "$stem.vcf.gz" "$arg"
		python /c/archaic/src/pyscripts/simplify_vcf.py "$stem.vcf.gz" GT
		rm "$stem.vcf.gz"
		bgzip "$stem.vcf"
		bcftools index "$stem.vcf.gz"
		
	fi
	let i++
done
