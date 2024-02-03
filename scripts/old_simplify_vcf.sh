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
	python $FILENAME ${multi[@]}
	bgzip -f $FILESTEM
	bcftools index $FILENAME
done

