#!/bin/bash
#PBS -P nf33
#PBS -q normal
#PBS -l walltime=02:00:00
#PBS -l mem=64GB
#PBS -l ncpus=8
#PBS -l storage=gdata/nf33
#PBS -l wd
#PBS -j oe

echo "Job started"
date

/home/561/ls3248/miniconda3/bin/python -u threshold.py 2>&1 | tee -a realtime.log

echo "Job finished"
date