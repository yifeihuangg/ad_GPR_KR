#!/bin/bash
#SBATCH --job-name=adp_compare
#SBATCH --output=adp_%A_%a.out
#SBATCH --error=adp_%A_%a.err
#SBATCH --array=1-400
#SBATCH --partition=sandyb
#SBATCH --ntasks=1


######################
# Begin work section #
######################

python adp_comp_parallel.py $SLURM_ARRAY_TASK_ID
