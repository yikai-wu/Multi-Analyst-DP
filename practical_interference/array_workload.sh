#!/bin/bash
#SBATCH -t 90:00:00
#SBATCH -o ./sbatchlog/array-%A-%a.out
#SBATCH -c 4
#SBATCH --mem=32G
#SBATCH --partition=compsci
#SBATCH --array=11-20
python3 race_vary.py -k=20 -t=10 -ty=4 -seed=$SLURM_ARRAY_TASK_ID -run=$SLURM_ARRAY_TASK_ID