#!/bin/bash

#SBATCH --job-name=renyiadapt
#SBATCH --account=qc_group
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=jfurches@vt.edu

#SBATCH --partition=normal_q
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=16GB
#SBATCH --time=1-00:00:00
#SBATCH --export=ALL
#SBATCH --array=1-10%4

module load site/tinkercliffs/easybuild/setup
module load Julia/1.9.3-linux-x86_64

# Create environment
# julia --project=. -e "import Pkg; Pkg.instantiate()"

JULIA_NUM_THREADS=1
OPENBLAS_NUM_THREADS=4

julia --project=. performance.jl $SLURM_ARRAY_TASK_ID
