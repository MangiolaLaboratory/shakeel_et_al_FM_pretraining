from pathlib import Path
import shutil
import anndata as ad
import numpy as np
import pandas as pd
import torch
import json
import pickle
import argparse
import session_info
import os
os.chdir('/hpcfs/groups/phoenix-hpc-mangiola_laboratory/haroon/geneformer/gene_perturb_task/original_code/')

parser = argparse.ArgumentParser(description='Run GeneFormer')
parser.add_argument('--dataset_name', dest='dataset_name', action='store', required = True, help='The id of a file in output/results')
parser.add_argument('--aggregation_level', dest='aggregation_level', action='store', required = True, help='The aggregation level of the Geneformer model to use (e.g., metacell_2, metacell_4, subsample_2, subsample_4 etc.)')

parser.add_argument('--test_train_config_id', dest = 'test_train_config_id', action = 'store', required = True, help = "The ID of the test/train/holdout run")
parser.add_argument('--perturbation_type', dest = 'perturbation_type', action = 'store', required = True, help = "The Geneformer perturbation type ('delete' or 'overexpress'). For more information see: https://geneformer.readthedocs.io/en/latest/geneformer.in_silico_perturber.html")
parser.add_argument('--ncells_training', dest = 'ncells_training', action = 'store', default = 300 ,type = int)
parser.add_argument('--seed', dest = 'seed', action = 'store', help = "The seed of the run", default = 1, type = int)

parser.add_argument("--working_dir", dest = "working_dir", action='store', required = True, help = "The directory that contains the params, results, scripts etc.")
parser.add_argument("--result_id", dest = "result_id", action='store', required = True, help = "The result_id")

args = parser.parse_args()
# args = parser.parse_args(["--dataset_name", "norman_from_scfoundation",
#     "--perturbation_type", "overexpress",
#     "--test_train_config_id", "ede703be5dd78-c8355c044d3ba", "--working_dir",
#     "/scratch/ahlmanne/perturbation_prediction_benchmark", "--result_id", "0"])
print(args)
print("GPU: " + str(torch.cuda.get_device_name()))

freeze_entire_model = False
freeze_layers = 10

frozen_postfix  = "full_frozen" if freeze_entire_model else "half_frozen"

MODEL_VARIANT = "104M"  # Specify the model variant, e.g., "30M" or "104M"
emb_dims = 256 if MODEL_VARIANT == "30M" else 768


BASE_TMP = f"/hpcfs/groups/phoenix-hpc-mangiola_laboratory/haroon/geneformer/gene_perturb_task/original_code/tmp_{MODEL_VARIANT}/"


out_dir = args.working_dir + "/results" + "_" + args.perturbation_type + "_perturbation_" + MODEL_VARIANT + "_" + frozen_postfix + "/" + args.aggregation_level + "/"  + args.dataset_name + "/" + args.result_id + "/"
tmp_out_dir = os.path.join(BASE_TMP, args.aggregation_level, args.dataset_name,args.result_id)

if not os.path.exists(out_dir):
    os.makedirs(out_dir, exist_ok=True)

if not os.path.exists(tmp_out_dir):
    os.makedirs(tmp_out_dir, exist_ok=True)

if MODEL_VARIANT == "104M":  
  MODEL_PATHS = {
                # "metacell_62500": "/g/data/wd17/haroon/geneformer/trained_foundation_models/models/104M_AGGmetacell_62500_12_emb768_SL4096_E3_B12_LR0.0002_LScosine_WU0_Oadamw/final_trained_model/metacell_62500",
                # "metacell_125000": "/g/data/wd17/haroon/geneformer/trained_foundation_models/models/104M_AGGmetacell_125000_12_emb768_SL4096_E3_B12_LR0.0002_LScosine_WU0_Oadamw/final_trained_model/metacell_125000",
                # "metacell_250000": "/g/data/wd17/haroon/geneformer/trained_foundation_models/models/104M_AGGmetacell_250000_12_emb768_SL4096_E3_B12_LR0.0002_LScosine_WU0_Oadamw/final_trained_model/metacell_250000",
                # "metacell_500000": "/g/data/wd17/haroon/geneformer/trained_foundation_models/models/104M_AGGmetacell_500000_12_emb768_SL4096_E3_B12_LR0.0002_LScosine_WU0_Oadamw/final_trained_model/metacell_500000",
                # "metacell_1000000": "/g/data/wd17/haroon/geneformer/trained_foundation_models/models/104M_AGGmetacell_1000000_12_emb768_SL4096_E3_B12_LR0.0002_LScosine_WU0_Oadamw/final_trained_model/metacell_1000000",
                # "subsample_62500": "/g/data/wd17/haroon/geneformer/trained_foundation_models/models/104M_AGGsubsample_62500_12_emb768_SL4096_E3_B12_LR0.0002_LScosine_WU0_Oadamw/final_trained_model/subsample_62500",
                # "subsample_125000": "/g/data/wd17/haroon/geneformer/trained_foundation_models/models/104M_AGGsubsample_125000_12_emb768_SL4096_E3_B12_LR0.0002_LScosine_WU0_Oadamw/final_trained_model/subsample_125000",
                # "subsample_250000": "/g/data/wd17/haroon/geneformer/trained_foundation_models/models/104M_AGGsubsample_250000_12_emb768_SL4096_E3_B12_LR0.0002_LScosine_WU0_Oadamw/final_trained_model/subsample_250000",
                # "subsample_500000": "/g/data/wd17/haroon/geneformer/trained_foundation_models/models/104M_AGGsubsample_500000_12_emb768_SL4096_E3_B12_LR0.0002_LScosine_WU0_Oadamw/final_trained_model/subsample_500000",
                # "subsample_1000000": "/g/data/wd17/haroon/geneformer/trained_foundation_models/models/104M_AGGsubsample_1000000_12_emb768_SL4096_E3_B12_LR0.0002_LScosine_WU0_Oadamw/final_trained_model/subsample_1000000",
                "subsample_2000000": "/hpcfs/groups/phoenix-hpc-mangiola_laboratory/haroon/geneformer/trained_foundation_models/models/104M_AGGsubsample_2000000_12_emb768_SL4096_E3_B12_LR0.0002_LScosine_WU0_Oadamw/final_trained_model/subsample_2000000",
                "subsample_4000000": "/hpcfs/groups/phoenix-hpc-mangiola_laboratory/haroon/geneformer/trained_foundation_models/models/104M_AGGsubsample_4000000_12_emb768_SL4096_E3_B12_LR0.0002_LScosine_WU0_Oadamw/final_trained_model/subsample_4000000",
                "subsample_8000000": "/hpcfs/groups/phoenix-hpc-mangiola_laboratory/haroon/geneformer/trained_foundation_models/models/104M_AGGsubsample_8000000_12_emb768_SL4096_E3_B12_LR0.0002_LScosine_WU0_Oadamw/final_trained_model/subsample_8000000",
                "subsample_16000000": "/hpcfs/groups/phoenix-hpc-mangiola_laboratory/haroon/geneformer/trained_foundation_models/models/104M_AGGsubsample_16000000_12_emb768_SL4096_E3_B12_LR0.0002_LScosine_WU0_Oadamw/final_trained_model/subsample_16000000",
                "genecorpus": "/hpcfs/groups/phoenix-hpc-mangiola_laboratory/haroon/geneformer/trained_foundation_models/models/104M_AGGgenecorpus_12_emb768_SL4096_E3_B12_LR0.0002_LScosine_WU0_Oadamw/final_trained_model/genecorpus",
                "singlecell": "/hpcfs/groups/phoenix-hpc-mangiola_laboratory/haroon/geneformer/trained_foundation_models/models/104M_AGGsinglecell_12_emb768_SL4096_E3_B12_LR0.0002_LScosine_WU0_Oadamw/final_trained_model/singlecell",
                "random_62500": "/hpcfs/groups/phoenix-hpc-mangiola_laboratory/haroon/geneformer/trained_foundation_models/models/104M_AGGrandom_62500_12_emb768_SL4096_E3_B12_LR0.0002_LScosine_WU0_Oadamw/final_trained_model/random_62500",
                "random_125000": "/hpcfs/groups/phoenix-hpc-mangiola_laboratory/haroon/geneformer/trained_foundation_models/models/104M_AGGrandom_125000_12_emb768_SL4096_E3_B12_LR0.0002_LScosine_WU0_Oadamw/final_trained_model/random_125000",
                "random_250000": "/hpcfs/groups/phoenix-hpc-mangiola_laboratory/haroon/geneformer/trained_foundation_models/models/104M_AGGrandom_250000_12_emb768_SL4096_E3_B12_LR0.0002_LScosine_WU0_Oadamw/final_trained_model/random_250000",
                "random_500000": "/hpcfs/groups/phoenix-hpc-mangiola_laboratory/haroon/geneformer/trained_foundation_models/models/104M_AGGrandom_500000_12_emb768_SL4096_E3_B12_LR0.0002_LScosine_WU0_Oadamw/final_trained_model/random_500000",
                "random_1000000": "/hpcfs/groups/phoenix-hpc-mangiola_laboratory/haroon/geneformer/trained_foundation_models/models/104M_AGGrandom_1000000_12_emb768_SL4096_E3_B12_LR0.0002_LScosine_WU0_Oadamw/final_trained_model/random_1000000",
                "random_2000000": "/hpcfs/groups/phoenix-hpc-mangiola_laboratory/haroon/geneformer/trained_foundation_models/models/104M_AGGrandom_2000000_12_emb768_SL4096_E3_B12_LR0.0002_LScosine_WU0_Oadamw/final_trained_model/random_2000000",
                "random_4000000": "/hpcfs/groups/phoenix-hpc-mangiola_laboratory/haroon/geneformer/trained_foundation_models/models/104M_AGGrandom_4000000_12_emb768_SL4096_E3_B12_LR0.0002_LScosine_WU0_Oadamw/final_trained_model/random_4000000",
                "random_8000000": "/hpcfs/groups/phoenix-hpc-mangiola_laboratory/haroon/geneformer/trained_foundation_models/models/104M_AGGrandom_8000000_12_emb768_SL4096_E3_B12_LR0.0002_LScosine_WU0_Oadamw/final_trained_model/random_8000000",
                "random_16000000": "/hpcfs/groups/phoenix-hpc-mangiola_laboratory/haroon/geneformer/trained_foundation_models/models/104M_AGGrandom_16000000_12_emb768_SL4096_E3_B12_LR0.0002_LScosine_WU0_Oadamw/final_trained_model/random_16000000",
                
                }

elif MODEL_VARIANT == "30M":
  MODEL_PATHS = {
                # "metacell_62500": "/g/data/wd17/haroon/geneformer/trained_foundation_models/models/30M_AGGmetacell_62500_6_emb256_SL2048_E2_B12_LR0.001_LSlinear_WU10000_Oadamw/final_trained_model/metacell_62500",
                # "metacell_125000": "/g/data/wd17/haroon/geneformer/trained_foundation_models/models/30M_AGGmetacell_125000_6_emb256_SL2048_E2_B12_LR0.001_LSlinear_WU10000_Oadamw/final_trained_model/metacell_125000",
                # "metacell_250000": "/g/data/wd17/haroon/geneformer/trained_foundation_models/models/30M_AGGmetacell_250000_6_emb256_SL2048_E2_B12_LR0.001_LSlinear_WU10000_Oadamw/final_trained_model/metacell_250000",
                # "metacell_500000": "/g/data/wd17/haroon/geneformer/trained_foundation_models/models/30M_AGGmetacell_500000_6_emb256_SL2048_E2_B12_LR0.001_LSlinear_WU10000_Oadamw/final_trained_model/metacell_500000",
                # "metacell_1000000": "/g/data/wd17/haroon/geneformer/trained_foundation_models/models/30M_AGGmetacell_1000000_6_emb256_SL2048_E2_B12_LR0.001_LSlinear_WU10000_Oadamw/final_trained_model/metacell_1000000",
                # "metacell_2000000": "/g/data/wd17/haroon/geneformer/trained_foundation_models/models/30M_AGGmetacell_2000000_6_emb256_SL2048_E2_B12_LR0.001_LSlinear_WU10000_Oadamw/final_trained_model/metacell_2000000",
                # "metacell_4000000": "/g/data/wd17/haroon/geneformer/trained_foundation_models/models/30M_AGGmetacell_4000000_6_emb256_SL2048_E2_B12_LR0.001_LSlinear_WU10000_Oadamw/final_trained_model/metacell_4000000",
                # "metacell_8000000": "/g/data/wd17/haroon/geneformer/trained_foundation_models/models/30M_AGGmetacell_8000000_6_emb256_SL2048_E2_B12_LR0.001_LSlinear_WU10000_Oadamw/final_trained_model/metacell_8000000",
                # "subsample_62500": "/g/data/wd17/haroon/geneformer/trained_foundation_models/models/30M_AGGsubsample_62500_6_emb256_SL2048_E2_B12_LR0.001_LSlinear_WU10000_Oadamw/final_trained_model/subsample_62500",
                # "subsample_125000": "/g/data/wd17/haroon/geneformer/trained_foundation_models/models/30M_AGGsubsample_125000_6_emb256_SL2048_E2_B12_LR0.001_LSlinear_WU10000_Oadamw/final_trained_model/subsample_125000",
                # "subsample_250000": "/g/data/wd17/haroon/geneformer/trained_foundation_models/models/30M_AGGsubsample_250000_6_emb256_SL2048_E2_B12_LR0.001_LSlinear_WU10000_Oadamw/final_trained_model/subsample_250000",
                # "subsample_500000": "/g/data/wd17/haroon/geneformer/trained_foundation_models/models/30M_AGGsubsample_500000_6_emb256_SL2048_E2_B12_LR0.001_LSlinear_WU10000_Oadamw/final_trained_model/subsample_500000",
                # "subsample_1000000": "/g/data/wd17/haroon/geneformer/trained_foundation_models/models/30M_AGGsubsample_1000000_6_emb256_SL2048_E2_B12_LR0.001_LSlinear_WU10000_Oadamw/final_trained_model/subsample_1000000",
                # "subsample_2000000": "/g/data/wd17/haroon/geneformer/trained_foundation_models/models/30M_AGGsubsample_2000000_6_emb256_SL2048_E2_B12_LR0.001_LSlinear_WU10000_Oadamw/final_trained_model/subsample_2000000",
                # "subsample_4000000": "/g/data/wd17/haroon/geneformer/trained_foundation_models/models/30M_AGGsubsample_4000000_6_emb256_SL2048_E2_B12_LR0.001_LSlinear_WU10000_Oadamw/final_trained_model/subsample_4000000",
                # "subsample_8000000": "/g/data/wd17/haroon/geneformer/trained_foundation_models/models/30M_AGGsubsample_8000000_6_emb256_SL2048_E2_B12_LR0.001_LSlinear_WU10000_Oadamw/final_trained_model/subsample_8000000",
                # "subsample_16000000": "/hpcfs/groups/phoenix-hpc-mangiola_laboratory/haroon/geneformer/trained_foundation_models/models/30M_AGGsubsample_16000000_6_emb256_SL2048_E2_B12_LR0.001_LSlinear_WU10000_Oadamw/final_trained_model/subsample_16000000",
                # "genecorpus": "/hpcfs/groups/phoenix-hpc-mangiola_laboratory/haroon/geneformer/trained_foundation_models/models/30M_AGGgenecorpus_6_emb256_SL2048_E2_B12_LR0.001_LSlinear_WU10000_Oadamw/final_trained_model/genecorpus"
                "singlecell": "/hpcfs/groups/phoenix-hpc-mangiola_laboratory/haroon/geneformer/trained_foundation_models/models/30M_AGGsinglecell_6_emb256_SL2048_E2_B12_LR0.001_LSlinear_WU10000_Oadamw/final_trained_model/singlecell",
                "metacell_16000000": "/hpcfs/groups/phoenix-hpc-mangiola_laboratory/haroon/geneformer/trained_foundation_models/models/30M_AGGmetacell_16000000_6_emb256_SL2048_E2_B12_LR0.001_LSlinear_WU10000_Oadamw/final_trained_model/metacell_16000000",
                "random_62500": "/hpcfs/groups/phoenix-hpc-mangiola_laboratory/haroon/geneformer/trained_foundation_models/models/30M_AGGrandom_62500_6_emb256_SL2048_E2_B12_LR0.001_LSlinear_WU10000_Oadamw/final_trained_model/random_62500",
                "random_125000": "/hpcfs/groups/phoenix-hpc-mangiola_laboratory/haroon/geneformer/trained_foundation_models/models/30M_AGGrandom_125000_6_emb256_SL2048_E2_B12_LR0.001_LSlinear_WU10000_Oadamw/final_trained_model/random_125000",
                "random_250000": "/hpcfs/groups/phoenix-hpc-mangiola_laboratory/haroon/geneformer/trained_foundation_models/models/30M_AGGrandom_250000_6_emb256_SL2048_E2_B12_LR0.001_LSlinear_WU10000_Oadamw/final_trained_model/random_250000",
                "random_500000": "/hpcfs/groups/phoenix-hpc-mangiola_laboratory/haroon/geneformer/trained_foundation_models/models/30M_AGGrandom_500000_6_emb256_SL2048_E2_B12_LR0.001_LSlinear_WU10000_Oadamw/final_trained_model/random_500000",
                "random_1000000": "/hpcfs/groups/phoenix-hpc-mangiola_laboratory/haroon/geneformer/trained_foundation_models/models/30M_AGGrandom_1000000_6_emb256_SL2048_E2_B12_LR0.001_LSlinear_WU10000_Oadamw/final_trained_model/random_1000000",
                "random_2000000": "/hpcfs/groups/phoenix-hpc-mangiola_laboratory/haroon/geneformer/trained_foundation_models/models/30M_AGGrandom_2000000_6_emb256_SL2048_E2_B12_LR0.001_LSlinear_WU10000_Oadamw/final_trained_model/random_2000000",
                "random_4000000": "/hpcfs/groups/phoenix-hpc-mangiola_laboratory/haroon/geneformer/trained_foundation_models/models/30M_AGGrandom_4000000_6_emb256_SL2048_E2_B12_LR0.001_LSlinear_WU10000_Oadamw/final_trained_model/random_4000000",
                "random_8000000": "/hpcfs/groups/phoenix-hpc-mangiola_laboratory/haroon/geneformer/trained_foundation_models/models/30M_AGGrandom_8000000_6_emb256_SL2048_E2_B12_LR0.001_LSlinear_WU10000_Oadamw/final_trained_model/random_8000000",
                "random_16000000": "/hpcfs/groups/phoenix-hpc-mangiola_laboratory/haroon/geneformer/trained_foundation_models/models/30M_AGGrandom_16000000_6_emb256_SL2048_E2_B12_LR0.001_LSlinear_WU10000_Oadamw/final_trained_model/random_16000000"
  }

else:
  raise ValueError("MODEL_VARIANT must be either '30M' or '104M'")



# GENEFORMER_MODEL_LOCATION = "/hpcfs/groups/phoenix-hpc-mangiola_laboratory/haroon/geneformer/trained_foundation_models/models/30M_AGGmetacell_128_6_emb256_SL2048_E2_B12_LR0.001_LSlinear_WU10000_Oadamw/final_trained_model/metacell_128"
GENEFORMER_MODEL_LOCATION = MODEL_PATHS[args.aggregation_level]

print(f"Using Geneformer model at: {GENEFORMER_MODEL_LOCATION}")

np.random.seed(args.seed)
# --------------------------------------------------------

if args.dataset_name == "norman_from_scfoundation":
  import sys
   # scfoundation uses a forked version of GEARS which
  sys.path.insert(0, "/g/huber/users/ahlmanne/projects/perturbation_prediction-benchmark/tmp/scfoundation/scfoundation_gears/")
  sys.path.append("/g/huber/users/ahlmanne/projects/perturbation_prediction-benchmark/tmp/scfoundation/model/")
  import gears.version
  assert gears.version.__version__ == '0.0.2'
else:
  import gears.version
  assert gears.version.__version__ == '0.1.2'

from gears import PertData, GEARS
from gears.utils import filter_pert_in_go


pert_data_folder = Path("data/" + args.dataset_name + "/gears_pert_data/")
pert_data = PertData(pert_data_folder)
if args.dataset_name in ['norman', 'adamson', 'dixit']:
  pert_data.load(args.dataset_name)
else:
  pert_data.load(data_path = "data/gears_pert_data/" + args.dataset_name)

adata = pert_data.adata
with open(args.working_dir + "/results" + "_" + args.perturbation_type + "_perturbation_" + MODEL_VARIANT + "_" + frozen_postfix + "/" + args.test_train_config_id) as json_file:
  set2conditions = json.load(json_file)



# Convert adata.X[ij] = log(c / (s / 10k) + 1) to counts
counts = np.expm1(adata.X) / 1e4
# The smallest non-zero element used to be a 1
one_equiv = np.array([np.min(counts[i,:].data) for i in range(counts.shape[0])])
counts = (counts / one_equiv[:,np.newaxis]).tocsr()
new_adata = ad.AnnData(counts, obs = adata.obs, var = adata.var)
new_adata.obs['n_counts'] = np.round(new_adata.X.sum(axis=1))

# with open("/hpcfs/groups/phoenix-hpc-mangiola_laboratory/haroon/geneformer/Geneformer/geneformer/gene_dictionaries_30m/ensembl_mapping_dict_gc30M.pkl", "rb") as f:
#   gene2ensembl = pickle.load(f)
with open("/hpcfs/groups/phoenix-hpc-mangiola_laboratory/haroon/geneformer/Geneformer/geneformer/ensembl_mapping_dict_gc104M.pkl", "rb") as f:
  gene2ensembl = pickle.load(f)

if args.dataset_name in ['adamson', 'norman', 'replogle_k562_essential', 'replogle_rpe1_essential']:
  # Already has ENSEMBL IDs
  new_adata.var['ensembl_id'] = new_adata.var.index
else:
  # Add ENSEMBL IDs 
  gene_info = pd.DataFrame.from_dict(gene2ensembl, orient = "index", columns = ["ensembl_id"])
  gene_info = gene_info.drop_duplicates(subset = "ensembl_id", keep = "first")
  if args.dataset_name == "dixit":
    new_adata.var = new_adata.var.set_index('gene_name', drop=False)
  
  new_adata.var = new_adata.var.join(gene_info, how = "left", validate = "one_to_one")
  new_adata = new_adata[:, new_adata.var.dropna().index]

print(f"Matched {new_adata.shape[1]} rows out of {adata.shape[1]}")

ctrl_adata = new_adata[new_adata.obs['condition'] == "ctrl",:].copy()
new_adata = new_adata[new_adata.obs['condition'].isin(set2conditions['train']), :]

# Geneformer
from geneformer import TranscriptomeTokenizer
from geneformer import Classifier

from geneformer import InSilicoPerturber
from geneformer import InSilicoPerturberStats
from geneformer import perturber_utils as pu


import tempfile
from datetime import datetime


trained_model_dir = os.path.join(BASE_TMP, "trained_model" + "_" + args.perturbation_type + "_perturbation/", args.aggregation_level, args.dataset_name)
if not os.path.exists(trained_model_dir):
    os.makedirs(trained_model_dir, exist_ok=True)


# Tokenization
def tokenize_adata(adata, output_dir = None, prefix = "tokenized"):
  # if not output_dir:
    # output_dir = tempfile.mkdtemp()
  # with  tempfile.TemporaryDirectory() as adata_dir:

  if output_dir is None:
    output_dir = os.path.join(BASE_TMP, args.dataset_name, "tokenized_" + args.dataset_name + "_" + str(prefix))
    os.makedirs(output_dir, exist_ok=True)
  adata_dir = os.path.join(BASE_TMP, args.dataset_name, "adata_" + args.dataset_name + "_" + str(prefix))
  os.makedirs(adata_dir, exist_ok=True)
  if not os.path.exists(adata_dir + "/adata.h5ad"):
    adata.write_h5ad(adata_dir + "/adata.h5ad")
  if not os.path.exists(f"{output_dir}/tokenized.dataset"):
    tk = TranscriptomeTokenizer(
        {"condition": "condition"},
        model_version="V2" if MODEL_VARIANT == "104M" else "V1",
        # nproc=16,
                                # chunk_size=512,
                                # model_version="V2",
                                # model_input_size=4096, # Max input size of model to truncate input to.
                                #                       # For the 30M model series, should be 2048. For the 95M model series, should be 4096.
                                # special_token=True, # For the 30M model series, should be False. For the 95M model series, should be True.
                                # collapse_gene_ids=True,
                                # gene_median_file="/hpcfs/groups/phoenix-hpc-mangiola_laboratory/haroon/geneformer/Geneformer/geneformer/gene_dictionaries_30m/gene_median_dictionary_gc30M.pkl", # Path to pickle file containing dictionary of non-zero median gene expression values across Genecorpus-30M.
                                # token_dictionary_file="/hpcfs/groups/phoenix-hpc-mangiola_laboratory/haroon/geneformer/Geneformer/geneformer/gene_dictionaries_30m/token_dictionary_gc30M.pkl", # Path to pickle file containing token dictionary (Ensembl IDs:token).
                                # gene_mapping_file="/hpcfs/groups/phoenix-hpc-mangiola_laboratory/haroon/geneformer/Geneformer/geneformer/gene_dictionaries_30m/ensembl_mapping_dict_gc30M.pkl", # Path to pickle file containing dictionary for collapsing gene IDs.
                                
                                )
  
    tk.tokenize_data(data_directory = adata_dir, 
                      output_directory = output_dir, 
                      output_prefix = "tokenized", 
                      file_format="h5ad")
  return f"{output_dir}/tokenized.dataset"

ctrl_adata_tok = tokenize_adata(ctrl_adata, output_dir=None, prefix = "ctrl_tokenized")
train_adata_tok = tokenize_adata(new_adata,output_dir=None, prefix = "train_tokenized")

# Important that the date is determined here as the training can take multiple days
today_date = datetime.now().strftime('%y%m%d')



# Classification (i.e., pretraining)
cc = Classifier(classifier="cell",
                cell_state_dict = {"state_key": "condition", "states": "all"},
                freeze_layers = freeze_layers,
                num_crossval_splits = 1,
                max_ncells_per_class = args.ncells_training,
                forward_batch_size=200 if MODEL_VARIANT == "30M" else 16,
                freeze_entire_model = freeze_entire_model,
                nproc=1)
cc.prepare_data(input_data_file = train_adata_tok,
                output_directory=trained_model_dir,
                output_prefix="classification",
                split_id_dict=None)
all_metrics = cc.validate(model_directory=GENEFORMER_MODEL_LOCATION,
                          prepared_input_data_file=f"{trained_model_dir}/classification_labeled_train.dataset",
                          id_class_dict_file=f"{trained_model_dir}/classification_id_class_dict.pkl",
                          output_directory=trained_model_dir,
                          output_prefix="classification_result")

trained_model_location = f"{trained_model_dir}/{today_date}_geneformer_cellClassifier_classification_result/ksplit1"





# In Silico Perturbation

# isp.perturb_data internally calculates the embedding after perturbation, but there is no way to extract it
# So I copied the code from https://huggingface.co/ctheodoris/Geneformer/blob/main/geneformer/in_silico_perturber.py#L421
# and https://huggingface.co/ctheodoris/Geneformer/blob/main/geneformer/in_silico_perturber.py#L761
# to export the embedding
def get_perturbed_embedding(isp: InSilicoPerturber, model, input_data):
    from geneformer.emb_extractor import get_embs

    isp.max_len = pu.get_model_input_size(model)
    layer_to_quant = pu.quant_layers(model) + isp.emb_layer

    ### filter input data ###
    # general filtering of input data based on filter_data argument


    def make_group_perturbation_batch(example):
        example_input_ids = example["input_ids"]
        example["tokens_to_perturb"] = isp.tokens_to_perturb
        indices_to_perturb = [
            example_input_ids.index(token) if token in example_input_ids else None
            for token in isp.tokens_to_perturb
        ]
        indices_to_perturb = [
            item for item in indices_to_perturb if item is not None
        ]
        if len(indices_to_perturb) > 0:
            example["perturb_index"] = indices_to_perturb
        else:
            # -100 indicates tokens to overexpress are not present in rank value encoding
            example["perturb_index"] = [-100]
        if isp.perturb_type == "delete":
            example = pu.delete_indices(example)
        elif isp.perturb_type == "overexpress":
            example = pu.overexpress_tokens(
                example, isp.max_len, isp.special_token
            )
            example["n_overflow"] = pu.calc_n_overflow(
                isp.max_len,
                example["length"],
                isp.tokens_to_perturb,
                indices_to_perturb,
            )
        return example


    perturbed_data = input_data.map(
        make_group_perturbation_batch, num_proc=isp.nproc
    ) 

    if isp.perturb_type == "overexpress":
        input_data = input_data.add_column(
            "n_overflow", perturbed_data["n_overflow"]
        )
        input_data = input_data.map(
            pu.truncate_by_n_overflow_special, num_proc=isp.nproc
        )


    perturbation_cls_emb = get_embs(
      model,
      perturbed_data,
      "cls" if MODEL_VARIANT == "104M" else "cell",
      layer_to_quant,
      isp.pad_token_id,
      isp.forward_batch_size,
      token_gene_dict=isp.token_gene_dict,
      summary_stat=None,
      silent=True,
    )
    return perturbation_cls_emb.cpu().detach().numpy()



model = pu.load_model(
    "CellClassifier", num_classes = len(new_adata.obs['condition'].values.categories),
    model_directory = trained_model_location, mode="eval"
)
ctrl_dataset = pu.load_and_filter(
    None, 1, ctrl_adata_tok
)
start_isp = InSilicoPerturber(perturb_type=args.perturbation_type,
                      model_type="CellClassifier",
                      emb_mode="cls" if MODEL_VARIANT == "104M" else "cell",
                      model_version="V2" if MODEL_VARIANT == "104M" else "V1",
                      # emb_mode="cell",
                      max_ncells=150,
                      emb_layer=0,
                      forward_batch_size=100 if MODEL_VARIANT == "30M" else 50,
                      # token_dictionary_file="/hpcfs/groups/phoenix-hpc-mangiola_laboratory/haroon/geneformer/Geneformer/geneformer/gene_dictionaries_30m/token_dictionary_gc30M.pkl",
                      nproc=1) 
ctrl_dataset = start_isp.apply_additional_filters(ctrl_dataset)

conds = adata.obs["condition"].cat.remove_unused_categories().cat.categories.tolist()
embeddings = {}
for idx, genes in enumerate(conds):
  gene_list = list(set(genes.split("+")) - set(["ctrl"]))
  print(gene_list)
  mappable_genes = [gene2ensembl[x] for x in gene_list if x in gene2ensembl.keys()]
  if genes == "ctrl":
    embeddings[genes] = np.zeros(emb_dims)
    print("CTRL detected")
  elif len(mappable_genes) == len(gene_list):
    print("Processing perturbation for genes: " + str(gene_list))
    isp = InSilicoPerturber(perturb_type=args.perturbation_type,
                          genes_to_perturb= mappable_genes,
                          model_type="CellClassifier",
                          emb_mode="cls" if MODEL_VARIANT == "104M" else "cell",
                          model_version="V2" if MODEL_VARIANT == "104M" else "V1",
                          max_ncells=150,
                          emb_layer=0,
                          forward_batch_size=100 if MODEL_VARIANT == "30M" else 50,
                          # token_dictionary_file="/hpcfs/groups/phoenix-hpc-mangiola_laboratory/haroon/geneformer/Geneformer/geneformer/gene_dictionaries_30m/token_dictionary_gc30M.pkl",
                          nproc=1) 
    local_emb = get_perturbed_embedding(isp, model = model, input_data = ctrl_dataset)
    embeddings[genes] = local_emb.mean(axis=0)
  else:
    pass

 
 
 
# Using the embedding for each perturbation, predict expression!

def mean_gene_expr_for_condition(condition):
    sub_data = adata[adata.obs['condition'] == condition,:]
    return np.array(sub_data.X.mean(axis=0)).flatten()


training_df = (pd.DataFrame({"training": set2conditions.keys(),
              "condition": set2conditions.values()})
  .explode("condition"))

base_df = (pd.DataFrame({
 "condition": conds})
  .merge(training_df, how = "left", on = "condition", validate = "one_to_one")
  .assign(obs_mean = lambda x: [mean_gene_expr_for_condition(y) for y in x.condition])
  .assign(embedding = lambda x: [embeddings[y] if y in embeddings.keys() else np.nan for y in x.condition])
)

base_df = base_df.dropna()

train = base_df[base_df['training'] == "train"]
from sklearn.linear_model import Ridge
expr_predictor = Ridge(fit_intercept = True)
Y = np.vstack(train['obs_mean'])
emb_mat_train = np.vstack(train['embedding'])
expr_predictor.fit(X = emb_mat_train, y = Y)


emb_mat = np.vstack(base_df['embedding'])
res = expr_predictor.predict(X = emb_mat)
all_pred_vals = {k : res[i,:].tolist() for i,k in enumerate(base_df['condition'])}



with open(f"{out_dir}/all_predictions.json", 'w', encoding="utf8") as handle:
    json.dump(all_pred_vals, handle, indent = 4)
with open(f"{out_dir}/gene_names.json", 'w', encoding="utf8") as handle:
    json.dump(adata.var["gene_name"].values.tolist(), handle, indent = 4)

# Move results to out_dir
# shutil.move(tmp_out_dir, out_dir)



session_info.show()
print("Python done")
