#!/bin/bash
#SBATCH -t 90:00:00
#SBATCH -o ./sbatchlog/array-%A-%a.out
#SBATCH -c 4
#SBATCH --mem=32G
#SBATCH --partition=compsci
#SBATCH --array=11-20
python3 oneway_marginal_vary.py -d=8 -k=20 -t=10 -seed=$SLURM_ARRAY_TASK_ID -run=$SLURM_ARRAY_TASK_ID