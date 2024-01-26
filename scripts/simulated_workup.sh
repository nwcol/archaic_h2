#!/bin/bash

# take .vcf files and prepare them for analysis by bgzipping, indexing,
# masking, and rebgzipping

# arg 1: path to .bed mask file
# additional args: .vcf files


mask=$1


i=0
for arg in "$@";
do
	if [ "$i" -ge 1 ]
	then
	  stem="${arg:0:-4}" # .vcf is 4 characters

		bgzip "$stem.vcf"
		bcftools index "$arg.vcf.gz"
		bcftools view -o "$stem.vcf" -R "$mask" "$stem.vcf.gz"
		rm "$stem.vcf.gz"
		rm "$stem.vcf.gz.csi"
		bgzip "$stem.vcf"
	fi
	let i++
done

_EOF_