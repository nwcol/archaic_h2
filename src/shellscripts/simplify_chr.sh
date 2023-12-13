#!/bin/bash

# Simplify .vcf.gz files and overwrite the originals. Keeps only the GT field

# args: .vcf.gz files to Simplify


source /c/anaconda3/etc/profile.d/co*nda.sh
conda activate archaic_conda_env


for arg in "$@";
do
	if ! test -f "$arg.csi";
	then
		bcftools index "$arg"
	fi
		
	stem="${arg:0:-7}" 
		
	python /c/archaic/src/pyscripts/simplify_vcf.py "$stem.vcf.gz" GT
	bgzip -f "$stem.vcf"
	rm "$arg.csi"
	bcftools index "$stem.vcf.gz"
done

_EOF_