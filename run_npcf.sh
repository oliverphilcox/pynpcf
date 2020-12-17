#!/bin/bash
#
#SBATCH --job-name=boss_cmass_npcf
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=ophilcox@princeton.edu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=05:59:59
#SBATCH --mem-per-cpu=8GB
#SBATCH --array=0-31
#SBATCH --output=/home/ophilcox/out/boss_cmass_npcf_%a.log

# NB: we run each computation on a single core since unparallelized

# Load modules
module load anaconda3
source activate ptenv

## User inputs
FILE_ROOT=boss_cmass
INPUT_DIR=/projects/QUIJOTE/Oliver/npcf

## Input files
SLICE_NO=$SLURM_ARRAY_TASK_ID
DmR_FILE="${INPUT_DIR}/${FILE_ROOT}_DmR${SLICE_NO}.txt"
R_FILE="${INPUT_DIR}/${FILE_ROOT}_R${SLICE_NO}.txt"

## Output strings
OUT_STR_DmR="${FILE_ROOT}_DmR${SLICE_NO}.txt"
OUT_STR_R="${FILE_ROOT}_R${SLICE_NO}.txt"

# Run D-R computation
echo "Computing (Data-Random) NPCF contributions from random slice ${SLICE_NO}"
python -u npcf_estimator.py $DmR_FILE $OUT_STR_DmR

# Run R computation
echo "Computing Random NPCF contributions from random slice ${SLICE_NO}"
python -u npcf_estimator.py $R_FILE $OUT_STR_R

echo "Computations complete!"
