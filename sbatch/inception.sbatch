#!/bin/bash

#SBATCH --job-name=A1_PP_DATA_t0-9_frozen_tanh_0_DarthJarJar

#SBATCH --output=/home/bjuarez/%j.out
#SBATCH --error=/home/bjuarez/%j.err

#SBATCH -A CS156b

# with preprocessing (40), without preprocessing (18)
#SBATCH -t 12:00:00

#SBATCH --ntasks=1

# with preprocessing (28 - max), without preprocessing (4)
#SBATCH --cpus-per-task=4

#SBATCH --mem=128G

#SBATCH --gres=gpu:1

#SBATCH --mail-user=bjuarez@caltech.edu

#SBATCH--mail-type=BEGIN
#SBATCH--mail-type=END
#SBATCH--mail-type=FAIL

module load cuda/10.2

source /home/bjuarez/venv/bin/activate

python /home/bjuarez/CS156b/scripts/inception.py simple frozen tanh 0 full use_PP_data