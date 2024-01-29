#!/bin/bash
#
# Give the contig to extract as the first argument (eg 22)

CONTIG=$1

for FILENAME in "${@:2}";
do
	if ! test -f "$FILENAME.csi";
	then
		bcftools index $FILENAME
	fi
	NEWFILENAME="chr${CONTIG}_${FILENAME}"
	bcftools view -r $CONTIG -o $NEWFILENAME $FILENAME
	bcftools index $NEWFILENAME
done
