#!/bin/bash


FILES=()
STARTDIR=$PWD
MINLENGTH=0


while [ $# -gt 0 ]; do
	case $1 in
		-o | --out)
			OUTPATH=$2
			shift
			;;
        -l | --min-length)
            MINLENGTH=$2
            shift
            ;;
		*)
			if [[ $1 == *.bed ]]
			then
				FILES+=($1)
			else
				echo "A non-.bed file format was passed as an argument: ${1}"
			fi
			;;
	esac
	shift
done


FULLPATHS=()


for FILENAME in ${FILES[@]};
do
	DIR=$(dirname $FILENAME)
	cd $DIR
	FILEPATH=$PWD
    FULLPATHS+="${FILEPATH}/$(basename $FILENAME) "
	cd $STARTDIR
done


python -m util.scripts.intersect_bed_files "$OUTPATH" "$MINLENGTH" "${FULLPATHS[@]}"

