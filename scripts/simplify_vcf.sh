#!/bin/bash

NFIELDS=1
while getopts "f:" opt; do
    case $opt in
        f) multi+=("$OPTARG");;
    esac
    let NFIELDS++
    let NFIELDS++
done

for FILENAME in ${@:$NFIELDS};
do
	FILESTEM="${FILENAME%.*}"
	python /home/nick/Projects/archaic/scripts/simplify_vcf.py $FILENAME ${multi[@]}
	bgzip -f $FILESTEM
	bcftools index $FILENAME
done

