import datetime

# imports
import os
import time

os.environ["NCCL_DEBUG"] = "INFO"
os.environ["OMPI_MCA_opal_cuda_support"] = "true"
os.environ["CONDA_OVERRIDE_GLIBC"] = "2.56"

import pickle
import random
import subprocess

import numpy as np
import pytz
import torch
from datasets import load_from_disk
from transformers import BertConfig, BertForMaskedLM, TrainingArguments

from geneformer import GeneformerPretrainer

import os
import glob

seed_num = 0
random.seed(seed_num)
np.random.seed(seed_num)
seed_val = 42
torch.manual_seed(seed_val)
torch.cuda.manual_seed_all(seed_val)

BASE_DIR = "/hpcfs/groups/phoenix-hpc-mangiola_laboratory/haroon"
AGGREGATION_LEVEL = "subsample_16000000"  # Set aggregation level to 'singlecell' for single-cell data
NUM_EXAMPLES_ALL = {"subsample_16000000": 12_002_552, "subsample_1000000": 1_111_001, "subsample_2000000": 2_034_915, 
"subsample_4000000": 3_825_953, "subsample_8000000": 7_154_991, "subsample_64": 392_617, "subsample_128": 163_070}
MODEL_VARIANT = "30M"  # Specify the model variant, e.g., "30M" or "95M"
num_examples = NUM_EXAMPLES_ALL[AGGREGATION_LEVEL]
RESUME_FROM_CHECKPOINT = False

                         
if MODEL_VARIANT == "30M":
    GENE_MEDIAN_FILE= "Geneformer/geneformer/gene_dictionaries_30m/gene_median_dictionary_gc30M.pkl"
    TOKEN_DICTIONARY_FILE= "Geneformer/geneformer/gene_dictionaries_30m/token_dictionary_gc30M.pkl"
    ENSEMBL_MAPPING_FILE= "Geneformer/geneformer/gene_dictionaries_30m/ensembl_mapping_dict_gc30M.pkl"
elif MODEL_VARIANT == "95M":
    GENE_MEDIAN_FILE= "Geneformer/geneformer/gene_median_dictionary_gc95M.pkl"
    TOKEN_DICTIONARY_FILE= "Geneformer/geneformer/token_dictionary_gc95M.pkl"
    ENSEMBL_MAPPING_FILE= "Geneformer/geneformer/ensembl_mapping_dict_gc95M.pkl"
else:
    raise ValueError("MODEL_VARIANT must be either '30M' or '95M'")

tokenized_data_path = os.path.join("tokenized_" + str(MODEL_VARIANT), "cellnexus_" + str(AGGREGATION_LEVEL) + ".dataset")



# set local time/directories
rootdir = os.path.join(BASE_DIR, "geneformer", "trained_foundation_models")
os.makedirs(rootdir, exist_ok=True)

if MODEL_VARIANT == "30M":
    # set model parameters
    # model type
    # max input size
    max_input_size = 2**11  # 2048
    # number of layers
    num_layers = 6
    # number of attention heads
    num_attn_heads = 4
    # number of embedding dimensions
    num_embed_dim = 256
    # intermediate size
    intermed_size = num_embed_dim * 2
    # activation function
    activ_fn = "relu"
    # initializer range, layer norm, dropout
    initializer_range = 0.02
    layer_norm_eps = 1e-12
    attention_probs_dropout_prob = 0.02
    hidden_dropout_prob = 0.02


    # set training parameters
    # total number of examples in Genecorpus-30M after QC filtering:
    # num_examples = 27_406_208
    # number gpus
    # batch size for training and eval
    geneformer_batch_size = 12
    # max learning rate
    max_lr = 1e-3
    # learning schedule
    lr_schedule_fn = "linear"
    # warmup steps
    warmup_steps = 10_000
    # number of epochs
    epochs = 2
    # optimizer
    optimizer = "adamw"
    # weight_decay
    weight_decay = 0.001


# output directories
run_name = f"{MODEL_VARIANT}_AGG{AGGREGATION_LEVEL}_{num_layers}_emb{num_embed_dim}_SL{max_input_size}_E{epochs}_B{geneformer_batch_size}_LR{max_lr}_LS{lr_schedule_fn}_WU{warmup_steps}_O{optimizer}"
training_output_dir = f"{rootdir}/models/{run_name}/"
logging_dir = f"{rootdir}/runs/{run_name}/"
model_output_dir = os.path.join(training_output_dir, "final_trained_model", AGGREGATION_LEVEL)


# ensure not overwriting previously saved model
model_output_file = os.path.join(model_output_dir, "pytorch_model.bin")
if os.path.isfile(model_output_file) is True:
    raise Exception("Model already saved to this directory.")



os.makedirs(training_output_dir, exist_ok=True)
os.makedirs(model_output_dir, exist_ok=True)
os.makedirs(logging_dir, exist_ok=True)

# load gene_ensembl_id:token dictionary (e.g. https://huggingface.co/datasets/ctheodoris/Genecorpus-30M/blob/main/token_dictionary.pkl)
with open(TOKEN_DICTIONARY_FILE, "rb") as fp:
    token_dictionary = pickle.load(fp)






if MODEL_VARIANT == "30M":

    # model configuration
    config = {
        "hidden_size": 256,
        "num_hidden_layers": 6,
        "initializer_range": 0.02,
        "layer_norm_eps": 1e-12,
        "attention_probs_dropout_prob": 0.02,
        "hidden_dropout_prob": 0.02,
        "intermediate_size": 512,
        "hidden_act": "relu",
        "max_position_embeddings": 2048,
        "model_type": "bert",
        "num_attention_heads": 4,
        "pad_token_id": 0, #token_dictionary.get("<pad>"),
        "vocab_size": 25426, #len(token_dictionary),  # genes+2 for <mask> and <pad> tokens
    }

config = BertConfig(**config)
model = BertForMaskedLM(config)
model = model.train()
geneformer_batch_size = 12


# Find the most recent checkpoint
def find_latest_checkpoint(output_dir):
    checkpoints = glob.glob(os.path.join(output_dir, "checkpoint-*"))
    if not checkpoints:
        return None
    # Sort by checkpoint number and return the latest
    latest = max(checkpoints, key=lambda x: int(x.split("-")[-1]))
    return latest

# Get the latest checkpoint
latest_checkpoint = find_latest_checkpoint(training_output_dir)

if MODEL_VARIANT == "30M":

    # define the training arguments
    training_args = {
        "learning_rate": 1e-3,
        "do_train": True,
        "do_eval": False,
        "group_by_length": True,
        "length_column_name": "length",
        "disable_tqdm": False,
        "lr_scheduler_type": "linear",
        "warmup_steps": 10_000,
        "weight_decay": 0.001,
        "per_device_train_batch_size": geneformer_batch_size,
        "num_train_epochs": 3,
        "save_strategy": "steps",
        "save_steps": 25000, # np.floor(num_examples / geneformer_batch_size / 8),  # 8 saves per epoch
        "logging_steps": 1000,
        "output_dir": training_output_dir,
        "logging_dir": logging_dir,
        # "resume_from_checkpoint": latest_checkpoint,  # Specific checkpoint path
        # "resume_from_checkpoint": RESUME_TRAINING,  # Add this line to resume training from previous checkpoint
    }

    # Only add resume_from_checkpoint if we have a checkpoint
# if RESUME_FROM_CHECKPOINT and latest_checkpoint is not None:
#     print(f"Resuming training from checkpoint: {latest_checkpoint}")
#     training_args["resume_from_checkpoint"] = latest_checkpoint

training_args = TrainingArguments(**training_args)

print("Starting training.")
start_time = time.time()
lengths_file_path = str(AGGREGATION_LEVEL) + "_" + str(MODEL_VARIANT) + "_" + "example_lengths.pkl"

# tokenized_data_path = "/hpcfs/users/a1236507/SAiGENCI/CellNexus/gc-30M_sample50k.dataset"

# define the trainer
trainer = GeneformerPretrainer(
    model=model,
    args=training_args,
    # pretraining corpus (e.g. https://huggingface.co/datasets/ctheodoris/Genecorpus-30M/tree/main/genecorpus_30M_2048.dataset)
    train_dataset=load_from_disk(tokenized_data_path),
    # file of lengths of each example cell (e.g. https://huggingface.co/datasets/ctheodoris/Genecorpus-30M/blob/main/genecorpus_30M_2048_lengths.pkl)
    example_lengths_file=lengths_file_path,
    token_dictionary=token_dictionary,
)

    # Only add resume_from_checkpoint if we have a checkpoint
if RESUME_FROM_CHECKPOINT and latest_checkpoint is not None:
    # train
    print(f"Resuming training from checkpoint: {latest_checkpoint}")
    trainer.train(resume_from_checkpoint = latest_checkpoint)
    
    # training_args["resume_from_checkpoint"] = latest_checkpoint
else:
    print("Starting training from scratch.")
    # train
    trainer.train()

end_time = time.time()
total_time = end_time - start_time

# Format time as hours, minutes, seconds
hours, remainder = divmod(total_time, 3600)
minutes, seconds = divmod(remainder, 60)
formatted_time = f"{int(hours)}h {int(minutes)}m {int(seconds)}s"


train_time_file_path = str(AGGREGATION_LEVEL) + "_" + str(MODEL_VARIANT) + "_" + "training_time.txt"

# Write the total training time to a file
with open(train_time_file_path, "w") as f:
    f.write(f"Total training time: {formatted_time}\n")
    

print(f"Training time recorded: {formatted_time}")


# save model
trainer.save_model(model_output_dir)

print(f"Model saved in {model_output_dir}")
