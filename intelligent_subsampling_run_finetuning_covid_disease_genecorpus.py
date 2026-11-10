import os

# Check current directory
print("Current directory:", os.getcwd())

# Change to your code directory (replace with your actual path)
os.chdir('/hpcfs/groups/phoenix-hpc-mangiola_laboratory/haroon/full_frozen_geneformer/Geneformer/geneformer')

# Verify the change
print("New directory:", os.getcwd())

import os
import sys

# Change to the repository root
os.chdir("/scratchdata1/groups/phoenix-hpc-mangiola_laboratory/haroon/full_frozen_geneformer/Geneformer")

# Add current directory to Python path
if os.getcwd() not in sys.path:
    sys.path.insert(0, os.getcwd())

from geneformer.finetuner import FineTuner

from geneformer.finetuner_utils import get_train_valid_test_splits

import json
import pickle


task_config = {
        "tasks": {
        "disease_classification": [
            # "genecorpus_heart_disease", 
            # "cellnexus_blood_disease", 
            "cellnexus_covid_disease"
            ],
        # "dosage_sensitivity": ["genecorpus_dosage_sensitivity"],
        # "cell_type_classification": ["cellnexus_cell_types"]
        },
        "aggregation_levels": [
           "genecorpus"
            ]
    }

# aggregation_level="metacell_32"
# task="dosage_sensitivity"
# dataset="genecorpus_dosage_sensitivity"
# aggregation_level="metacell_32"
# task="dosage_sensitivity"
# dataset="genecorpus_dosage_sensitivity"
model_version="V2"
base_dir= "/hpcfs/groups/phoenix-hpc-mangiola_laboratory/haroon/geneformer"
model_variant="104M"
crossval_splits = 5 # 1 or 5
freeze_num_encoder_layers=10
freeze_entire_model=False
freeze_tag = "full_frozen" if freeze_entire_model else "half_frozen"

    # Loop over tasks
for task, datasets in task_config["tasks"].items():
    # Loop over each dataset associated with the task
    for dataset in datasets:
        # Loop over aggregation levels
        for aggregation_level in task_config["aggregation_levels"]:
            print(f"Running task={task}, dataset={dataset}, aggregation_level={aggregation_level}")

            if task not in ["dosage_sensitivity", "cell_type_classification"]:
                crossval_split_metrics = {}

                for crossval_split in range(1, crossval_splits + 1):
                    print(f"Running cross-validation split {crossval_split} of {crossval_splits} for task={task}, dataset={dataset}, aggregation_level={aggregation_level}")

                    


                    if model_variant == "104M":
                        training_args = {
                        "num_train_epochs": 2,
                        "learning_rate": 0.000355,
                        "lr_scheduler_type": "cosine",
                        "warmup_steps": 500,
                        "weight_decay":0.05,
                        "per_device_train_batch_size": 8,
                        "seed": 73,
                        "evaluation_strategy":"epoch",        # Evaluate every epoch
                        "save_strategy":"epoch",              # Save checkpoint every epoch
                        "metric_for_best_model":"eval_loss",  # Metric to determine "best" model # Doc: https://huggingface.co/transformers/v3.5.1/main_classes/trainer.html#:~:text=after%20each%20evaluation.-,metric_for_best_model,-(str%2C
                        "greater_is_better":False,            # For loss, lower is better
                        "load_best_model_at_end":True,        # KEY: Load best model at the end
                        "save_total_limit":3,                 # Keep only 3 best checkpoints
                        # "logging_dir": os.path.normpath("D:/geneformer_finetuning/trained_cell_classification_models/disease_classification/genecorpus_heart_disease/30M_metacell_8/250623_geneformer_cellClassifier_genecorpus_heart_disease_test/ksplit1/runs"),
                        }
                    else:
                        training_args = {
                        "num_train_epochs": 10,
                        "learning_rate": 0.000804,
                        "lr_scheduler_type": "polynomial",
                        "warmup_steps": 1812,
                        "weight_decay":0.258828,
                        "per_device_train_batch_size": 64,
                        "seed": 73,
                        "evaluation_strategy":"epoch",        # Evaluate every epoch
                        "save_strategy":"epoch",              # Save checkpoint every epoch
                        "metric_for_best_model":"eval_loss",  # Metric to determine "best" model # Doc: https://huggingface.co/transformers/v3.5.1/main_classes/trainer.html#:~:text=after%20each%20evaluation.-,metric_for_best_model,-(str%2C
                        "greater_is_better":False,            # For loss, lower is better
                        "load_best_model_at_end":True,        # KEY: Load best model at the end
                        "save_total_limit":3,                 # Keep only 3 best checkpoints
                        # "logging_dir": os.path.normpath("D:/geneformer_finetuning/trained_cell_classification_models/disease_classification/genecorpus_heart_disease/30M_metacell_8/250623_geneformer_cellClassifier_genecorpus_heart_disease_test/ksplit1/runs"),
                        }
                    

                    input_data_file, cell_state_dict, filter_data_dict, train_test_id_split_dict, train_valid_id_split_dict = get_train_valid_test_splits(
                        TASK=task,
                        DATASET=dataset,
                        MODEL_VARIANT=model_variant,
                        DATASET_PATH = os.path.join(base_dir, "datasets", task, dataset),
                        CROSSVAL_SPLITS=crossval_splits,
                    )



                    finetuner = FineTuner(base_dir=base_dir,
                        aggregation_level=aggregation_level,
                        model_variant=model_variant,
                        task=task,
                        dataset=dataset,
                        crossval_splits=crossval_splits,
                        freeze_tag=freeze_tag)
                    

                    
                    output_prefix = "test" if crossval_splits == 1 else "ksplit" + str(crossval_split)

                    model_dir = os.path.join(finetuner.output_dir, f"geneformer_cellClassifier_{output_prefix}")
                    existing_metrics_path = os.path.join(model_dir, f"{output_prefix}_eval_metrics_dict.pkl")
                    
                    if os.path.exists(existing_metrics_path):
                        print(f"Loading existing metrics for cross-validation split {crossval_split} from {existing_metrics_path}")
                        # FIXED: Variable name consistency
                        with open(existing_metrics_path, "rb") as f:
                            all_metrics = pickle.load(f)

                    else:
        
                        all_metrics = finetuner.finetune_model(
                            training_args=training_args, 
                            cell_state_dict = cell_state_dict, 
                            filter_data_dict = filter_data_dict, 
                            input_data_file = input_data_file, 
                            output_prefix = output_prefix, 
                            train_test_id_split_dict = train_test_id_split_dict[str(crossval_split)] if train_test_id_split_dict is not None else None, 
                            train_valid_id_split_dict = train_valid_id_split_dict[str(crossval_split)] if train_valid_id_split_dict is not None else None, 
                            num_crossval_splits= 1 if task not in ["dosage_sensitivity", "cell_type_classification"] else crossval_splits,
                            freeze_num_encoder_layers=freeze_num_encoder_layers,
                            freeze_entire_model=freeze_entire_model,
                            split_sizes={"train": 0.8, "valid": 0.2, "test": 0.0}
                            )
                    
                    crossval_split_metrics[str(crossval_split)] = all_metrics

                metrics_path = os.path.join(finetuner.output_dir, "metrics.pkl")
                with open(metrics_path, "wb") as f:
                    pickle.dump(crossval_split_metrics, f)

                print(f"✅ Metrics saved to: {metrics_path}")

            elif task in ["dosage_sensitivity", "cell_type_classification"]:

                print(f"Running cross-validation for task={task}, dataset={dataset}, aggregation_level={aggregation_level}")

                



                if model_variant == "104M":
                    training_args = {
                    "num_train_epochs": 2,
                    "learning_rate": 0.000355,
                    "lr_scheduler_type": "cosine",
                    "warmup_steps": 500,
                    "weight_decay":0.05,
                    "per_device_train_batch_size": 8,
                    "seed": 73,
                    "evaluation_strategy":"epoch",        # Evaluate every epoch
                    "save_strategy":"epoch",              # Save checkpoint every epoch
                    "metric_for_best_model":"eval_loss",  # Metric to determine "best" model # Doc: https://huggingface.co/transformers/v3.5.1/main_classes/trainer.html#:~:text=after%20each%20evaluation.-,metric_for_best_model,-(str%2C
                    "greater_is_better":False,            # For loss, lower is better
                    "load_best_model_at_end":True,        # KEY: Load best model at the end
                    "save_total_limit":3,                 # Keep only 3 best checkpoints
                    # "logging_dir": os.path.normpath("D:/geneformer_finetuning/trained_cell_classification_models/disease_classification/genecorpus_heart_disease/30M_metacell_8/250623_geneformer_cellClassifier_genecorpus_heart_disease_test/ksplit1/runs"),
                    }
                else:
                    training_args = {
                    "num_train_epochs": 10,
                    "learning_rate": 0.000804,
                    "lr_scheduler_type": "polynomial",
                    "warmup_steps": 1812,
                    "weight_decay":0.258828,
                    "per_device_train_batch_size": 64,
                    "seed": 73,
                    "evaluation_strategy":"epoch",        # Evaluate every epoch
                    "save_strategy":"epoch",              # Save checkpoint every epoch
                    "metric_for_best_model":"eval_loss",  # Metric to determine "best" model # Doc: https://huggingface.co/transformers/v3.5.1/main_classes/trainer.html#:~:text=after%20each%20evaluation.-,metric_for_best_model,-(str%2C
                    "greater_is_better":False,            # For loss, lower is better
                    "load_best_model_at_end":True,        # KEY: Load best model at the end
                    "save_total_limit":3,                 # Keep only 3 best checkpoints
                    # "logging_dir": os.path.normpath("D:/geneformer_finetuning/trained_cell_classification_models/disease_classification/genecorpus_heart_disease/30M_metacell_8/250623_geneformer_cellClassifier_genecorpus_heart_disease_test/ksplit1/runs"),
                    }
                

                input_data_file, cell_state_dict, filter_data_dict, train_test_id_split_dict, train_valid_id_split_dict = get_train_valid_test_splits(
                    TASK=task,
                    DATASET=dataset,
                    MODEL_VARIANT=model_variant,
                    DATASET_PATH = os.path.join(base_dir, "datasets", task, dataset)
,
                    CROSSVAL_SPLITS=crossval_splits,
                )



                finetuner = FineTuner(base_dir=base_dir,
                    aggregation_level=aggregation_level,
                    model_variant=model_variant,
                    task=task,
                    dataset=dataset,
                    crossval_splits=crossval_splits,
                    freeze_tag=freeze_tag)
                
                output_prefix = "5fold" if crossval_splits == 5 else "test"

                metrics_path = os.path.join(finetuner.output_dir, "metrics.pkl")
                if os.path.exists(metrics_path):
                    print(f"Loading existing metrics for cross-validation splits from {metrics_path}")
                    with open(metrics_path, "rb") as f:
                        all_metrics = pickle.load(f)

                else:
                    print(f"Running finetuning for task={task}, dataset={dataset}, aggregation_level={aggregation_level}")
                
                
    
                    all_metrics = finetuner.finetune_model(
                        training_args=training_args, 
                        cell_state_dict = cell_state_dict, 
                        filter_data_dict = filter_data_dict, 
                        input_data_file = input_data_file, 
                        output_prefix = output_prefix, 
                        train_test_id_split_dict = train_test_id_split_dict[str(crossval_split)] if train_test_id_split_dict is not None else None, 
                        train_valid_id_split_dict = train_valid_id_split_dict[str(crossval_split)] if train_valid_id_split_dict is not None else None, 
                        num_crossval_splits= 1 if task not in ["dosage_sensitivity", "cell_type_classification"] else crossval_splits,
                        freeze_num_encoder_layers=freeze_num_encoder_layers,
                        freeze_entire_model=freeze_entire_model,
                        split_sizes={"train": 0.8, "valid": 0.2, "test": 0.0}
                        )
                    

                    with open(metrics_path, "wb") as f:
                        pickle.dump(all_metrics, f)

                    print(f"✅ Metrics saved to: {metrics_path}")