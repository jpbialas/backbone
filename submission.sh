#!/bin/bash
#PBS -q mediumq
#PBS -l select=1:ncpus=14
#PBS -l walltime=8:00:00
#PBS -e errors
#PBS -j eo
#PBS -J 0-8

# NOTE
# '#PBS' directives must immediately follow your shell initialization line '#!/bin/<shell>'
# '#PBS' directives must be consecutively listed without any empty lines in-between directives
# Reference the PBS Pro User Guide for #PBS directive options.
# To determine the appropriate queue (#PBS -q) and walltime (#PBS -l walltime) to use,
#  run (qmgr -c 'print server') on the login node.

# This is an example MPI job script for the JPL Zodiac cluster

# Set the output directory
# By default, PBS copies stdout and stderr files back to $PBS_O_WORKDIR
# When 'qsub' is run, PBS sets $PBS_O_WORKDIR to the directory where qsub is run.
# Change this environment variable if desired
#
#export PBS_O_WORKDIR=/home/jsfrank/backbone

# Set your executable directory (optional)
#
export RUN_DIR=/home/jsfrank/backbone

# Load software modules
# Available modules can be found with the command 'module avail'
# You must first load the appropriate init script to load modules environment variable
# Module init scripts are located at /usr/share/modules/init/ on Zodiac
#
source /usr/share/modules/init/bash
module load compilers/intel-11.1.069_64
module load mpi-sgi/2.04_64

#Run your application
#
cd $RUN_DIR
#mpiexec -n 4 ./executable
python al_haiti.py ${PBS_ARRAY_INDEX} rf model