# Code to run Multimodal Experiments

To run the multimodal experiments, the following scripts are essential:

- `models.py`: Defines all model components, including the tabular encoder (FTTransformer), image encoders (MedViT, MedMamba, FetalCLIP, MedSigLip), fusion strategies and the final multimodal model that combines tabular features, image features, and the fusion MLP. To select the image encoder, comment/uncomment the desired encoder inside the **build_multimodal_models** function. Each encoder allows training from scratch or loading pretrained weights via the **pretrained_path** parameter.
- `utils.py`: Contains utility functions for training and validation loops, model evaluation, metric computation, saving metrics and predictions and generating plots (confusion matrix, ROC curve, training curves)
- `data.py`: Implements tabular preprocessing and PyTorch dataset classes for multimodal experiments, including normalization, categorical encoding, and image loading with TrivialAugmentWide data augmentation.
- `aggregate_metrics.py`: Reads metric JSON files from experiment result folders and aggregates them into a single Excel file, summarizing best and final model performance across runs and folds.
- `columns_config.json`: Configuration file that maps dataset columns into num_cols (numerical features), cat_cols (categorical features) and image_cols (image paths columns). 

## Internal Validation

Use the `train_validate_multi_seeds_retrospective.py` file to run multiple experimental trials with different random seeds using internal cross-validation folds from the retrospective dataset. For each run and fold, it saves metrics, training history, confusion matrices, ROC curves, and both best and final model weights. Uses predefined cross-validation splits.

## External Validation

Use the `external_validation.py` file to train the multimodal model on the full retrospective dataset and evaluates it on the prospective dataset. Saves final model weights, metrics, training history, confusion matrix, and ROC curve.

## Clinical Practice

Use `evaluate_clinical_practice.py` to normalize clinical practice variables and compute the same evaluation metrics used for the models, enabling direct comparison. Supports two decision thresholds for positive class prediction: > 0.5 or >= 0.5.

## Other scripts

- `evaluate_predict.py`: Loads a trained model and runs inference on a dataset, generating a file with predictions and probability scores
- `analyse_results_prospective.py`: Compares model predictions with clinical practice predictions on the prospective dataset and generates alignment scatter plots with correlation statistics for different correctness groups.


## Ensembles Experiment

This folder contains scripts to combine predictions from multimodal models and clinical practice.

- `ensembles_PC_multimodal.py`: Builds ensemble predictions combining a multimodal model and clinical practice predictions using mean and max voting strategies.


