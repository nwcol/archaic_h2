#!/bin/bash

# Add one or more chromosomes to an existing merge

# argument 1: output name
# argument 2: merged .vcf.gz
# argument 3: .bed mask file
# additional args: files to Add


out=$1
out_stem="${out:0:-7}" 
base=$2
base_stem="${base:0:-7}" 
mask_file=$3


if ! test -f "$base.csi";
then
	bcftools index "$base"
fi


i=0
for arg in "$@";
do
	if [ "$i" -ge 3 ]
	then
		if ! test -f "$arg.csi";
		then
			bcftools index "$arg"
		fi
		
		stem="${arg:0:-7}" 
		
		bcftools view -o "$stem.vcf" -R "$mask_file" "$arg"
		bgzip -f "$stem.vcf"
		bcftools index "$stem.vcf.gz"
		
	fi
	let i++
done


bcftools merge -o "$out" -0 "$base" "${@:4}"
