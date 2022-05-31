#!/bin/bash

#SBATCH --job-name=preprocessing_simple_solution_data_DarthJarJar

#SBATCH --output=/home/bjuarez/%j.out
#SBATCH --error=/home/bjuarez/%j.err

#SBATCH -A CS156b

#SBATCH -t 5:00:00

#SBATCH --ntasks=1

#SBATCH --cpus-per-task=2

#SBATCH --mem=128G

#SBATCH --gres=gpu:1

#SBATCH --mail-user=bjuarez@caltech.edu

#SBATCH--mail-type=BEGIN
#SBATCH--mail-type=END
#SBATCH--mail-type=FAIL

module load cuda/10.2

source /home/bjuarez/venv/bin/activate

python /home/bjuarez/CS156b/scripts/preprocessing.py simple 0 0 15000 test_only