#!/bin/bash
#
# Give the path to a .bed mask file as the first argument

BEDFILE=$1

for FILENAME in "${@:2}";
do 
	ZIPPEDNAME="${FILENAME}.gz"
	bgzip $FILENAME
	bcftools index $ZIPPEDNAME
	bcftools view -o $FILENAME -R $BEDFILE $ZIPPEDNAME
	bgzip -f $FILENAME
	echo "${FILENAME} completed"
done

