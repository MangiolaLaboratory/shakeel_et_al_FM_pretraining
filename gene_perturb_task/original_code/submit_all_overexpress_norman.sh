#!/usr/bin/env bash
set -euo pipefail

WD="/hpcfs/groups/phoenix-hpc-mangiola_laboratory/haroon/geneformer/gene_perturb_task/original_code"
LOGDIR="${WD}/logs"
# mkdir -p "${LOGDIR}"

# Shared args (override by exporting before calling bash)
DATASET_NAME="${DATASET_NAME:-norman}"
SEED="${SEED:-1}"
WORKING_DIR="${WORKING_DIR:-working_dir}"
PERTURBATION_TYPE="${PERTURBATION_TYPE:-overexpress}"
NCELLS_TRAINING="${NCELLS_TRAINING:-300}"

# Build TEST_TRAIN_CONFIG_ID from dataset/seed unless provided
TEST_TRAIN_CONFIG_ID="${TEST_TRAIN_CONFIG_ID:-seed_${SEED}_${DATASET_NAME}_split}"

AGG_LEVELS=(
  # metacell_62500
  # metacell_125000
  # metacell_250000
  # metacell_500000
  # subsample_62500
  # subsample_125000
  # subsample_250000
  # subsample_500000
  # metacell_1000000
  # subsample_1000000
  # metacell_2000000
  # subsample_2000000
  # metacell_4000000
  # subsample_4000000
  # metacell_8000000
  # subsample_8000000
  # genecorpus
  # subsample_2000000
  # subsample_4000000
  # subsample_16000000
  # singlecell
  # metacell_16000000
  # random_62500
  # random_125000
  # random_250000
  # random_500000
  # random_1000000
  # random_2000000
  # random_4000000
  random_8000000
  random_16000000
)

for AGG in "${AGG_LEVELS[@]}"; do
  RESULT_ID="${AGG}_seed${SEED}_results"
  JOB_NAME="genef-${AGG}"

  echo "Submitting: ${AGG} -> ${RESULT_ID}"

  sbatch \
    --job-name="genef-${AGG}" \
    --output="${LOGDIR}/${PERTURBATION_TYPE}_${AGG}_%j.out" \
    --export=ALL,AGG="${AGG}",RESULT_ID="${RESULT_ID}",DATASET_NAME="${DATASET_NAME}",TEST_TRAIN_CONFIG_ID="${TEST_TRAIN_CONFIG_ID}",WORKING_DIR="${WORKING_DIR}",PERTURBATION_TYPE="${PERTURBATION_TYPE}",NCELLS_TRAINING="${NCELLS_TRAINING}",SEED="${SEED}",LOGDIR="${LOGDIR}" \
    "${WD}/run_geneformer_single.sbatch"
done
