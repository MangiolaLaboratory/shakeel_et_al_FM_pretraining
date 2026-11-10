#!/usr/bin/bash
# SBATCH --account=saigencir001
# SBATCH --partition=saigenci_highmem
#SBATCH --account=saigencir003
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --qos=long
#SBATCH --time=120:00:00
#SBATCH --mem=250G
#SBATCH --job-name=singleDrug_Finetune
#SBATCH --output=/hpcfs/groups/phoenix-hpc-mangiola_laboratory/haroon/full_frozen_geneformer/Geneformer/geneformer/slurm_logs_intellligent_subsampling_finetuning/%x_%j.out
#SBATCH --error=/hpcfs/groups/phoenix-hpc-mangiola_laboratory/haroon/full_frozen_geneformer/Geneformer/geneformer/slurm_logs_intellligent_subsampling_finetuning/%x_%j.err
#SBATCH -p a100
#SBATCH --gres=gpu:4
#SBATCH --export=all

echo Running job
echo
# Organising job output
# SBATCH --output=/hpcfs/users/a1236507/SAiGENCI/CellNexus/SLURM_JOBID.out
# Load modules
module load CUDAcompat/12.2-535.161.08
source /home/a1236507/anaconda3/etc/profile.d/conda.sh
conda activate geneformer
echo
export HF_HOME='/hpcfs/groups/phoenix-hpc-mangiola_laboratory/haroon/geneformer/cache'
cd /hpcfs/groups/phoenix-hpc-mangiola_laboratory/haroon/full_frozen_geneformer/Geneformer/geneformer
echo Running script
echo
python intelligent_subsampling_run_finetuning_drug_dosage_singlecell.py
echo Job Finished
echo "$SLURM_JOBID on $HOSTNAME at $(date)"

