#!/bin/bash

#SBATCH --job-name=preprocess_imputation_DarthJarJar

#SBATCH --output=/home/mcgee/%j.out
#SBATCH --error=/home/mcgee/%j.err

#SBATCH -A CS156b

# can set to 8 to 10 hours to start to test
#SBATCH -t 10:00:00

# change if we use multiple srun tasks
#SBATCH --ntasks=1

# tune this if we use matlab or mathematica or r
#SBATCH --cpus-per-task=1

# bump this up if we start running out of memory
#SBATCH --mem=64G

# definitely want to use GPU acceleration of whatever library we do
#SBATCH --gres=gpu:1

#SBATCH --mail-user=mcgee@caltech.edu

#SBATCH--mail-type=BEGIN
#SBATCH--mail-type=END
#SBATCH--mail-type=FAIL

module load cuda/10.2

source /home/mcgee/venv/bin/activate

python /home/mcgee/CS156b/scripts/preprocess_edge_detection.py
