#PBS -P nf33
#PBS -q normal
#PBS -l walltime=01:00:00
#PBS -l ncpus=16
#PBS -l mem=64GB
#PBS -l wd
#PBS -l storage=gdata/nf33+scratch/nf33+gdata/xp65+gdata/gb02+gdata/ob53
#PBS -W umask=0022
#PBS -j oe

set -eu
module use /g/data/xp65/public/modules
module use /g/data/gb02/public/modules
module load conda/analysis3
module load dask_setup

python calc_drought_probability.py
