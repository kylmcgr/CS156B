#!/bin/bash

#SBATCH --job-name=CNN_test_DarthJarJar

#SBATCH --output=/home/kmcgraw/%j.out
#SBATCH --error=/home/kmcgraw/%j.err

#SBATCH -A CS156b

# can set to 8 to 10 hours to start to test
#SBATCH -t 1:30:00

# change if we use multiple srun tasks
#SBATCH --ntasks=1

# tune this if we use matlab or mathematica or r
#SBATCH --cpus-per-task=1

# bump this up if we start running out of memory
#SBATCH --mem=32G

# definitely want to use GPU acceleration of whatever library we do
#SBATCH --gres=gpu:1

#SBATCH --mail-user=kmcgraw@caltech.edu

#SBATCH--mail-type=BEGIN
#SBATCH--mail-type=END
#SBATCH--mail-type=FAIL

module load cuda/11.1
module load gcc/11.2.0

source /home/kmcgraw/venvs/cs156b/bin/activate

cd /central/groups/CS156b/
python my_script.py
