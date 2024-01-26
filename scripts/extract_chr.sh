#!/bin/bash

# Parse one chromosome out of a .vcf.gz using bcftools, simplify it,
# and apply a mask

# arg 1: contig/chromosome to extract
# additional args: .vcf.gz to extract from


source /c/anaconda3/etc/profile.d/conda.sh
conda activate archaic_conda_env

chr=$1

i=0
for arg in "$@";
do
	if [ "$i" -ge 1 ]
	then
		if ! test -f "$arg.csi";
		then
			bcftools index "$arg"
		fi
		
		stem=${arg:0:-7} # .vcf.gz is 7 characters
		bcftools view -r "$1" -o chr"$chr"_"$stem".vcf.gz $arg
	fi
	let i++
done
