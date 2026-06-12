# Final Model - Code & Usage Guide 

This folder contains the code to run the final multimodal architecture for mode of delivery prediction.

## Scripts

- `*final_model*`: Calls the multimodal model, extracts and concatenates embeddings into a final multimodal vector, reconstructs the best pipeline found with EDCA, applies feature selection, and outputs predictions.

- `*evaluate_model*`: Evaluates the final model on the full prospective dataset.

## Requirements

Before running, make sure you have the following:


|Requirement  | Details |
|--|--|
|Best Multimodal Model Weights  | Download from [weights](https://drive.google.com/drive/folders/1_MAX3CrnvtrBHTU_izK3f_6LHKpMugcv? |
|EDCA pipeline  | `*utils/final_pipeline_edca*` |
|Architecture files  | `*data.py*`, `*columns_config.json*`, `*models.py*` inside `*multimodal_architecture/*` |
|MedViT-V2 repository  | Must be placed inside `*multimodal_architecture/*` |
|Environments  | `*medvit2_full*` - see `*environments/*` for requirements |


## Setup

1. Clone or download this repository.
2. Set up the `*medvit2_full*` conda environment using the .yml file in `*environments/*` .
3. Place the MedViT-V2 repository inside `*multimodal_architecture/*` .
4. Download the model weights from the link above and place them in `*multimodal_architecture/models_prospective/MedViT2_nopt/*`.
5. Ensure `*data.py*`, `*columns_config.json*`, `*models.py*` are present in `*multimodal_architecture/*`.
6. Make sure you have a prospective dataset inside  `*datasets/*`
7. Run  `*evaluate_model*` to get the predictions for mode of delivery prediction on the prospective dataset. 

## Architecture Overview

The diagram below illustrates the proposed multimodal architecture workflow:

![final_architecture](utils/pipeline_model_interface.png) 