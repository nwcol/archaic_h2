#!/bin/bash


FILES=()
STARTDIR=$PWD
OUTDIR="."
INFO=0
ID=0
FILTER=0
QUAL=0


while [ $# -gt 0 ]; do
	case $1 in
		-f | --format)
			FORMAT=$2
			shift
			;;
		-i | --info)
			INFO=$2
			shift
			;;
		-o | --out)
			OUTDIR=$2
			shift
			;;
		---keep_id)
			ID=1
			;;
		--keep_filter)
			FILTER=1
			;;
		--keep_quality)
			QUAL=1
			;;
		*)
			if [[ $1 == *.vcf.gz ]]
			then
				FILES+=($1)
			else
				echo "A non-.vcf.gz file format was passed as an argument: ${1}"
			fi
			;;
	esac
	shift
done


for FILENAME in ${FILES[@]};
do
	DIR=$(dirname $FILENAME)
	cd $DIR
	INPATH=$PWD
	cd $STARTDIR
	cd $OUTDIR
	OUTPATH=$PWD
	STEM="$(basename ${FILENAME%%.*})"
	IN="${INPATH}/${STEM}.vcf.gz"
	OUT="${OUTPATH}/simplified_${STEM}.vcf"
	python -m util.scripts.simplify_vcf_file $IN $OUT $FORMAT $INFO $ID $FILTER $QUAL
	bgzip -f $OUT
	cd $STARTDIR
done

