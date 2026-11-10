#!/usr/bin/env bash
#SBATCH --account=saigencir003
#SBATCH --partition=a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=48:00:00
#SBATCH --mem=250G
#SBATCH --gres=gpu:4
#SBATCH --export=ALL
# leave job-name/output to submitter

module load CUDAcompat/12.2-535.161.08
source /home/a1236507/anaconda3/etc/profile.d/conda.sh
conda activate geneformer

set -euo pipefail

# Required env vars are passed via --export from submitter:
# AGG, RESULT_ID, DATASET_NAME, TEST_TRAIN_CONFIG_ID, WORKING_DIR, PERTURBATION_TYPE, NCELLS_TRAINING, SEED

export HF_HOME='/hpcfs/groups/phoenix-hpc-mangiola_laboratory/haroon/geneformer/cache'

cd /hpcfs/groups/phoenix-hpc-mangiola_laboratory/haroon/geneformer/gene_perturb_task/original_code/ || exit 1

echo "[$(date)] Start AGG=${AGG} RESULT_ID=${RESULT_ID} on ${HOSTNAME} (Job ${SLURM_JOB_ID})"


python run_geneformer.py \
  --dataset_name "${DATASET_NAME}" \
  --test_train_config_id "${TEST_TRAIN_CONFIG_ID}" \
  --working_dir "${WORKING_DIR}" \
  --result_id "${RESULT_ID}" \
  --aggregation_level "${AGG}" \
  --perturbation_type "${PERTURBATION_TYPE}" \
  --ncells_training "${NCELLS_TRAINING}" \
  --seed "${SEED}"

echo "[$(date)] Done AGG=${AGG} (Job ${PBS_JOBID})"
