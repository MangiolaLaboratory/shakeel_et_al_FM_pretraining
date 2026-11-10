# Re-evaluating the benefit of training scale in cell foundation models

Code and resources for reproducing benchmarks that evaluate the benefit of pretraining scale in single-cell foundation models.

> **Project status:** Code refactoring is in progress. Pretrained models and task-specific fine-tuning datasets are available for download and can be used with the task-specific fine-tuning scripts.

## Contents

- [Getting started](#getting-started)
- [Geneformer modifications](#geneformer-modifications)
- [Pretraining data](#pretraining-data)
- [Roadmap](#roadmap)
- [Citation](#citation)
- [Acknowledgments](#acknowledgments)

## Getting started

Create the environment using `requirements.txt` file.

### 1. Clone the repository

Clone this repository and open its root directory. Replace the placeholders below with the repository URL and directory name:

```bash
git clone https://github.com/MangiolaLaboratory/shakeel_et_al_FM_pretraining.git
cd shakeel_et_al_FM_pretraining
```

This repository includes a modified version of Geneformer to support freezing selected encoder layers or the entire encoder during fine-tuning. See [Geneformer modifications](#geneformer-modifications) for details, particularly if you use a separately installed copy of Geneformer.

### 2. Download models and fine-tuning datasets

Download the pretrained models and task-specific fine-tuning datasets from the following folder:

**[Download models and datasets from Google Drive](https://drive.google.com/drive/folders/1OGGofvHbc2br9cI1_1pmB1V9jDEnn3TT)**

Place the downloaded files in the corresponding folders within the cloned repository, replacing the empty placeholder folders.

> **Required:** The model and dataset folders in the repository are currently empty. Download and populate these folders before running fine-tuning.

### 3. Configure and run fine-tuning

1. Edit the relevant configuration files in the `CFG/` directory, including the paths to the downloaded models and datasets.
2. Run the Python fine-tuning script for the task you want to evaluate.
3. Complete fine-tuning for all five folds before generating the corresponding figures.

### 4. Generate figures

Once results are available for all five folds, run the corresponding Python script to generate the figures for that task.

## Geneformer modifications

The bundled Geneformer code adds support for controlling which parameters are trainable during fine-tuning. The remaining code follows the original implementation.

If you use a separately installed Geneformer library, apply the corresponding changes to its `Classifier` class and model-freezing logic.

### Additional classifier parameters

| Parameter | Type | Purpose |
| --- | --- | --- |
| `freeze_layers` | `int` | Number of initial encoder layers to freeze. |
| `freeze_entire_model` | `bool` | Freeze model parameters except the classifier and pooler. |

### Parameter-freezing logic

The implementation below freezes the requested encoder layers and reports the number of trainable and non-trainable parameters. When `freeze_entire_model` is enabled, it retains parameters whose names contain `classifier`, `head`, `fc`, `linear`, or `pooler`.

`def_freeze_layers` is the resolved layer-freezing value used by the surrounding classifier implementation; this snippet belongs within that implementation.

<details>
<summary>Show the parameter-freezing code</summary>

```python
if def_freeze_layers > 0:
    modules_to_freeze = model.bert.encoder.layer[:def_freeze_layers]
    for module in modules_to_freeze:
        for param in module.parameters():
            param.requires_grad = False

    trainable_params = sum(
        p.numel() for p in model.parameters() if p.requires_grad
    )
    non_trainable_params = sum(
        p.numel() for p in model.parameters() if not p.requires_grad
    )

    print(
        "Trainable parameters after freezing encoder layers: "
        f"{trainable_params:,}"
    )
    print(
        "Non-trainable parameters after freezing encoder layers: "
        f"{non_trainable_params:,}"
    )

if self.freeze_entire_model:
    # Keep parameters matching these names trainable.
    trainable_names = ["classifier", "head", "fc", "linear", "pooler"]

    for name, param in model.named_parameters():
        if any(keyword in name.lower() for keyword in trainable_names):
            print(f"Keeping trainable: {name}")
            param.requires_grad = True
        else:
            print(f"Freezing parameter: {name}")
            param.requires_grad = False

    trainable_params = sum(
        p.numel() for p in model.parameters() if p.requires_grad
    )
    non_trainable_params = sum(
        p.numel() for p in model.parameters() if not p.requires_grad
    )

    print(
        "Trainable parameters after freezing entire model: "
        f"{trainable_params:,}"
    )
    print(
        "Non-trainable parameters after freezing entire model: "
        f"{non_trainable_params:,}"
    )
```

</details>

## Pretraining data

Tokenized pretraining datasets are available for both the **10M** and **104M** model variants. The pretraining collection comprises **19 corpus versions**, ranging from approximately **266,000 to 34 million cells**.

Because the tokenized data occupy several hundred gigabytes, please contact the authors to arrange access.

## Perturbation prediction code


## Roadmap

- [ ] Create a single entry point for pretraining.
- [ ] Create example notebooks for fine-tuning.
- [ ] Create an example notebook for embedding-based network analysis.
- [ ] Create an example notebook for attention-based gene regulatory network (GRN) analysis.

## Citation

If you use this work, please cite:

```bibtex
@article{shakeel2026finetuning,
  title     = {Finetuning masking challenges narrow-task evaluation of cell foundation models},
  author    = {Shakeel, Muhammad Haroon and Shen, Mengyuan and Mangiola, Stefano},
  journal   = {bioRxiv},
  pages     = {2026--06},
  year      = {2026},
  publisher = {Cold Spring Harbor Laboratory}
}
```

## Acknowledgments

This repository includes modified Geneformer code. Credit for the original Geneformer implementation belongs to its authors. The modifications described above enable the layer-freezing settings used in this study.

Addtionally, this repository contains code for perturbation prediction made available by **[Ahlmann-Eltze et al.](https://github.com/const-ae/linear_perturbation_prediction-Paper/tree/main/benchmark)**
