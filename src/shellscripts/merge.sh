#!/bin/bash

# Take "RAW" eg unprocessed .vcf.gz files, simplify them, combine them, intersect them, apply a mask, and apply a lower bound to continuous region length

source /c/anaconda3/etc/profile.d/conda.sh
conda activate archaic_conda_env