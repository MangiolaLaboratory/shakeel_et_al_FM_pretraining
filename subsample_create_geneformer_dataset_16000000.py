from geneformer import TranscriptomeTokenizer
import os
import scanpy as sc
import numpy as np
import glob

from geneformer import TranscriptomeTokenizer
import os
import scanpy as sc
import numpy as np
import glob
import os
from datasets import load_from_disk
import pickle

def get_folder_size(path='.'):
    total_size = 0
    for dirpath, dirnames, filenames in os.walk(path):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            if os.path.isfile(fp):
                total_size += os.path.getsize(fp)
    return total_size

def sizeof_fmt(num, suffix="B"):
    for unit in ["", "K", "M", "G", "T", "P"]:
        if abs(num) < 1024.0:
            return f"{num:3.1f} {unit}{suffix}"
        num /= 1024.0
    return f"{num:.1f} P{suffix}"


AGGREGATION_LEVEL = "subsample_16000000"  # Set aggregation level to 'singlecell' for single-cell data
BASE_DIR = "/hpcfs/groups/phoenix-hpc-mangiola_laboratory/haroon/geneformer/intelligent_subsampling"
MODEL_VARIANT = "30M"  # Specify the model variant, e.g., "30M" or "95M"

CREATE_GENEFORMER_ANNDATA = True  # Set to True to create Geneformer AnnData files

input_anndata_dir = os.path.join(BASE_DIR, "anndata_" + str(AGGREGATION_LEVEL))
                         
genformer_anndata_dir = os.path.join(BASE_DIR, "geneformer_anndata_" + str(AGGREGATION_LEVEL))

# Create output directory if it doesn't exist
os.makedirs(genformer_anndata_dir, exist_ok=True)



# Process each file
if CREATE_GENEFORMER_ANNDATA:
    # Get all h5ad files in the input directory
    h5ad_files = glob.glob(os.path.join(input_anndata_dir, "*.h5ad"))

    print(f"Found {len(h5ad_files)} h5ad files to process")
    print("Creating Geneformer AnnData files...")
    for file_path in h5ad_files:
        try:
            # Get the filename without path
            filename = os.path.basename(file_path)
            print(f"Processing {filename}...")
            output_path = os.path.join(genformer_anndata_dir, filename)
            if os.path.exists(output_path):
                print(f"File {filename} already exists in output directory. Skipping...")
                continue
            
            # Load the AnnData object
            adata = sc.read_h5ad(file_path)
            
            # Add the specified attributes
            adata.var["ensembl_id"] = adata.var.index
            adata.obs["n_counts"] = np.array(adata.X.sum(axis=1)).flatten()
            adata.obs["joinid"] = list(range(adata.n_obs))
                        
            
            # Write modified AnnData to output folder
            adata.write(output_path)
            
            print(f"Saved modified file to {output_path}")
            
            # Clean up memory
            del adata
            
        except Exception as e:
            print(f"Error processing {file_path}: {str(e)}")

    print("Processing complete.")

if MODEL_VARIANT == "30M":
    GENE_MEDIAN_FILE= "Geneformer/geneformer/gene_dictionaries_30m/gene_median_dictionary_gc30M.pkl"
    TOKEN_DICTIONARY_FILE= "Geneformer/geneformer/gene_dictionaries_30m/token_dictionary_gc30M.pkl"
    ENSEMBL_MAPPING_FILE= "Geneformer/geneformer/gene_dictionaries_30m/ensembl_mapping_dict_gc30M.pkl"
elif MODEL_VARIANT == "104M":
    GENE_MEDIAN_FILE= "Geneformer/geneformer/gene_median_dictionary_gc104M.pkl"
    TOKEN_DICTIONARY_FILE= "Geneformer/geneformer/token_dictionary_gc104M.pkl"
    ENSEMBL_MAPPING_FILE= "Geneformer/geneformer/ensembl_mapping_dict_gc104M.pkl"
else:
    raise ValueError("MODEL_VARIANT must be either '30M' or '104M'")

if AGGREGATION_LEVEL == "singlecell":
    custom_attr_name_dict={"dataset_id":"dataset_id", "sample_id":"sample_id", "cell_type":"cell_type","disease": "disease", "sex":"sex", "is_immune":"is_immune", "self_reported_ethnicity":"self_reported_ethnicity" }

else:
    custom_attr_name_dict={"dataset_id":"dataset_id", "sample_id":"sample_id", "disease": "disease", "sex":"sex", "is_immune":"is_immune", "self_reported_ethnicity":"self_reported_ethnicity" }

custom_attr_name_dict = {"disease": "disease"}

tk = TranscriptomeTokenizer(custom_attr_name_dict=custom_attr_name_dict,
                            nproc=16,
                            chunk_size=512,
                            model_input_size=2048, # Max input size of model to truncate input to.
                                                   # For the 30M model series, should be 2048. For the 95M model series, should be 4096.
                            special_token=False if MODEL_VARIANT == "30M" else True, # For the 30M model series, should be False. For the 95M model series, should be True.
                            collapse_gene_ids=True,
                            gene_median_file=GENE_MEDIAN_FILE, # Path to pickle file containing dictionary of non-zero median gene expression values across Genecorpus-30M.
                            token_dictionary_file=TOKEN_DICTIONARY_FILE, # Path to pickle file containing token dictionary (Ensembl IDs:token).
                            gene_mapping_file=ENSEMBL_MAPPING_FILE, # Path to pickle file containing dictionary for collapsing gene IDs.
)

# genformer_anndata_dir = "/g/data/wd17/haroon/anndata_tmp"

tokenized_data_output_path = "tokenized_" + str(MODEL_VARIANT)
output_prefix = "cellnexus_" + str(AGGREGATION_LEVEL)
tk.tokenize_data(data_directory = genformer_anndata_dir, # Path to directory containing loom files or anndata files
                 output_directory = tokenized_data_output_path, # Path to directory where tokenized data will be saved as .dataset
                 output_prefix = output_prefix, # Prefix for output .dataset
                 file_format="h5ad", # Format of input files. Can be "loom" or "h5ad".
                 use_generator=True) # Whether to use generator or dict for tokenization


size_bytes = get_folder_size(tokenized_data_output_path)
print(f"Total size of '{tokenized_data_output_path}': {sizeof_fmt(size_bytes)}")

size_bytes = get_folder_size(genformer_anndata_dir)
print(f"Total size of '{genformer_anndata_dir}': {sizeof_fmt(size_bytes)}")


# Load the dataset
dataset = load_from_disk(os.path.join(tokenized_data_output_path, output_prefix + ".dataset"))

# If it's a DatasetDict (e.g., with 'train', 'test' splits)
if isinstance(dataset, dict) or hasattr(dataset, "keys"):
    for split_name, split_data in dataset.items():
        print(f"\nSplit: {split_name}")
        print("Columns:", split_data.column_names)
else:
    # It's a single Dataset
    print("Columns:", dataset.column_names)
    
print(f"Number of records: {len(dataset)}")
with open("dataset_info_" + str(AGGREGATION_LEVEL) + "_" + str(MODEL_VARIANT) + ".txt", "w") as f:
    f.write(f"Number of records: {len(dataset)}\n")
    f.write(f"Columns: {dataset.column_names}\n")

cellNexus_example_lengths = dataset['length']


lengths_file_path = str(AGGREGATION_LEVEL) + "_" + str(MODEL_VARIANT) + "_" + "example_lengths.pkl"
# Save to file
with open(lengths_file_path, "wb") as f:
    pickle.dump(cellNexus_example_lengths, f)

print(f"Example lengths saved to {lengths_file_path}")
