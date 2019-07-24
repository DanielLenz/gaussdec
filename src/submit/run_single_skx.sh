#!/bin/bash
#----------------------------------------------------
# Sample Slurm job script
#   for TACC Stampede2 SKX nodes
#
#   *** OpenMP Job on SKX Normal Queue ***
# 
# Last revised: 20 Oct 2017
#
# Notes:
#
#   -- Launch this script by executing
#   -- Copy/edit this script as desired.  Launch by executing
#      "sbatch skx.openmp.slurm" on a Stampede2 login node.
#
#   -- OpenMP codes run on a single node (upper case N = 1).
#        OpenMP ignores the value of lower case n,
#        but slurm needs a plausible value to schedule the job.
#
#   -- Default value of OMP_NUM_THREADS is 1; be sure to change it!
#
#   -- Increase thread count gradually while looking for optimal setting.
#        If there is sufficient memory available, the optimal setting
#        is often 48 (1 thread per core) but may be higher.

#----------------------------------------------------

#SBATCH -J myjob           # Job name
#SBATCH -o myjob.o%j       # Name of stdout output file
#SBATCH -e myjob.e%j       # Name of stderr error file
#SBATCH -p skx-normal      # Queue (partition) name
#SBATCH -N 1               # Total # of nodes (must be 1 for OpenMP)
#SBATCH -n 1               # Total # of mpi tasks (should be 1 for OpenMP)
#SBATCH -c 96              # Total number of cores
#SBATCH -t 08:00:00        # Run time (hh:mm:ss)
##SBATCH --mail-user=mail@daniellenz.org
##SBATCH --mail-type=all    # Send email at begin and end of job
##SBATCH -A myproject       # Allocation name (req'd if you have more than 1)

# Grab basepixel to be fit 
modeldir=$1
basepix=$2

if [[ -n "$basepix" ]]; then
    echo "basepix is $basepix"
else
    echo "Invalid basepix"
fi

# Get filenames
indices_filename=$WORK/projects/gaussdec/data/indices/indices$basepix.npy
out_filename=$modeldir/raw/basepix$basepix.h5
config_filename=$modeldir/decompose.yaml

# Set thread count (default value is 1)
export OMP_NUM_THREADS=96

# Launch OpenMP code
python3 $WORK/projects/gaussdec/src/decompose/call_specfit.py --config $config_filename -c True -x $indices_filename $out_filename

