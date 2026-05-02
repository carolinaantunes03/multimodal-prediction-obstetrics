
import sys
import os
import torch
import json
import pandas as pd
import numpy as np
from pathlib import Path
from torch.utils.data import DataLoader
import data
import models
import utils
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import random

# CHANGE THESE PATHS BEFORE RUNNING

train_data_path = "../datasets/retrospective-dataset/cv1/all_cv1_train.csv"     # Path to train split of one cross validation fold of the retrospective dataset
val_data_path = "../datasets/retrospective-dataset/cv1/all_cv1_validation.csv"  # Path to validation split of one cross validation fold of the retrospective dataset
test_data_path = "../datasets/retrospective-dataset/cv1/all_cv1_test.csv"   # Path to test split of one cross validation fold of the retrospective dataset
experiment_base = 'multi_seeds/exp_Name/cv1'    #Path to save the model and results

epochs = 100
patience = 15

seeds = [1, 6, 21, 10, 43, 50, 100, 336, 2025, 9999]


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", DEVICE)


EPOCHS = epochs
BATCH_SIZE = 16
WEIGHT_DECAY = 0.001
LEARNING_RATE = 0.000373345024672811
PATIENCE = patience

with open("columns_config.json", "r") as f:
        col_config = json.load(f)
num_cols = col_config["num_cols"]
cat_cols = col_config["cat_cols"]
image_cols = col_config["image_cols"]
y_col = "Class"

# Data Loading 
train_data = pd.read_csv(train_data_path)
val_data = pd.read_csv(val_data_path)
test_data = pd.read_csv(test_data_path)

train_data = train_data.drop('Processo', axis=1)
val_data = val_data.drop('Processo', axis=1)
test_data = test_data.drop('Processo', axis=1)

# Loop over all seeds       

for run_idx, seed in enumerate(seeds):
    print(f"\n{'='*40}\nStarting run {run_idx+1}/{len(seeds)} with seed {seed}\n{'='*40}")
    
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

   
    base_dir = Path(__file__).parent.resolve()
    run_name = f"exp_{experiment_base}_seed{seed}"
    model_dir   = base_dir / "models" / run_name
    results_dir = base_dir / "results" / run_name
    for p in [model_dir, results_dir]:
        p.mkdir(parents=True, exist_ok=True)
    
    BEST_MODEL_PATH  = model_dir / "best_model.pth"
    FINAL_MODEL_PATH = model_dir / "final_model.pth"
    HISTORY_FILE_PATH = results_dir / "train_history.json"
    CONFIG_FILE_PATH  = model_dir / "config_model.json"
    METRICS_BEST_MODEL  = results_dir / "test_metrics_best.json"
    METRICS_FINAL_MODEL = results_dir / "test_metrics_final.json"

    # data preprocessing 
    tab_preproc = data.TabularPreprocessor(num_cols, cat_cols)
    tab_preproc.fit(train_data)

    # image root auto-detection for debug 
    possible_roots = [
        "/home/beatrix/Documents/Carolina_A/tese-carolina/datasets/retrospective-dataset",   # path to images (for debug)
        "/home/carolantunes/datasets/retrospective-dataset" 
    ]

    image_root = None
    for root in possible_roots:
        candidate = os.path.join(root, "images/Abdomen")
        if os.path.exists(candidate):
            image_root = root
            break

    if image_root is None:
        raise FileNotFoundError(
        "   ould not find the images directory. Please check dataset paths."
    )
    else:
        print(f"Using image root: {image_root}")

    train_dataset = data.MultimodalDataset(train_data, tab_preproc, image_cols, y_col, train=True, root_dir=image_root)
    val_dataset   = data.MultimodalDataset(val_data, tab_preproc, image_cols, y_col, train=False, root_dir=image_root)
    test_dataset  = data.MultimodalDataset(test_data, tab_preproc, image_cols, y_col, train=False, root_dir=image_root)

    print("First 5 image_valid_num:", [train_dataset[i]["image_valid_num"] for i in range(5)])


    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader   = DataLoader(val_dataset, batch_size=16, shuffle=False)
    test_loader  = DataLoader(test_dataset, batch_size=16, shuffle=False)



    # model setup
    num_numerical = len(num_cols)
    num_categories = [train_data[col].nunique() + 1 for col in cat_cols]
    multimodal_model = models.build_multimodal_model(
        num_numerical=num_numerical,
        num_categories=num_categories,
        tabular_token_dim=192,
        tabular_hidden_dim=192
    ).to(DEVICE)

    optimizer = torch.optim.AdamW(multimodal_model.parameters(), lr=3.7e-4, weight_decay=1e-3)

    '''
    # --- optimizer setup with differential learning rates ---
    image_params = []
    other_params = []

    for name, param in multimodal_model.named_parameters():
        # Select MedSigLIP fine-tuned parameters only
        if "image_encoder.model" in name and param.requires_grad:
            image_params.append(param)
        else:
            other_params.append(param)

    optimizer = torch.optim.AdamW([
        {"params": other_params, "lr": 3.7e-4},  # fusion + tabular parts
        {"params": image_params, "lr": 5e-5},    # fine-tuned MedSigLIP last 2 blocks
    ], weight_decay=1e-3)

    print(f"Optimizer created with {len(image_params)} MedSigLIP fine-tuned parameter tensors "
      f"and {len(other_params)} frozen/other parameter tensors.")
    '''
    criterion = torch.nn.CrossEntropyLoss()

    utils.save_config_file(CONFIG_FILE_PATH, "AdamW", 3.7e-4, 16, epochs)

    # training
    
    results = utils.train_and_validate(
        multimodal_model, train_loader, val_loader, optimizer, criterion,
        DEVICE, BEST_MODEL_PATH, FINAL_MODEL_PATH, epochs, patience
    )
    

    utils.save_train_metrics(
        history_file_path = HISTORY_FILE_PATH,
        train_losses = results["train_losses"],
        val_losses   = results["val_losses"],
        train_accuracies = results["train_accuracies"],
        val_accuracies   = results["val_accuracies"],
        best_epoch = results["best_epoch"],
        elapsed_time = results ["elapsed_time"]
    )

    # test best model 
    best_model = models.build_multimodal_model(num_numerical, num_categories, 192, 192).to(DEVICE)
    best_model.load_state_dict(torch.load(BEST_MODEL_PATH))
    best_metrics = utils.evaluate(best_model, test_loader, DEVICE)
    utils.save_test_metrics(
        metrics_file_path = METRICS_BEST_MODEL,
        balanced_accuracy = best_metrics["balanced_accuracy"],
        acc = best_metrics["acc"],
        precision = best_metrics["precision"],
        recall = best_metrics["recall"],
        f1 = best_metrics["f1"],
        roc_auc = best_metrics["roc_auc"],
        matthews = best_metrics["matthews"],
        specificity = best_metrics["specificity"],
        tnr=best_metrics["tnr"]
    )

    CM_BEST_MODEL = results_dir / "conf_matrix_best.png"
    ROC_BEST_MODEL = results_dir / "roc_curve_best.png"
    class_names = ['Cesarean birth', 'Vaginal birth']

    utils.plot_confusionMatrix(
        CM_BEST_MODEL,
        best_metrics["confusion_matrix"],
        class_names,
        "Confusion Matrix (Best Model)"
    )
    utils.plot_RocCurve(
        ROC_BEST_MODEL,
        targets=best_metrics["targets"],
        prob_scores=best_metrics["prob_scores"]
    )

    # test final model 
    final_model = models.build_multimodal_model(num_numerical, num_categories, 192, 192).to(DEVICE)
    final_model.load_state_dict(torch.load(FINAL_MODEL_PATH))
    final_metrics = utils.evaluate(final_model, test_loader, DEVICE)
    utils.save_test_metrics(
        metrics_file_path = METRICS_FINAL_MODEL,
        balanced_accuracy = final_metrics["balanced_accuracy"],
        acc = final_metrics["acc"],
        precision = final_metrics["precision"],
        recall = final_metrics["recall"],
        f1 = final_metrics["f1"],
        roc_auc = final_metrics["roc_auc"],
        matthews = final_metrics["matthews"],
        specificity = final_metrics["specificity"],
        tnr=final_metrics["tnr"]
    )

    CM_FINAL_MODEL = results_dir / "conf_matrix_final.png"
    ROC_FINAL_MODEL = results_dir / "roc_curve_final.png"
    class_names = ['Cesarean birth', 'Vaginal birth']

    utils.plot_confusionMatrix(
        CM_FINAL_MODEL,
        final_metrics["confusion_matrix"],
        class_names,
        "Confusion Matrix (Final Model)"
    )
    utils.plot_RocCurve(
        ROC_FINAL_MODEL,
        targets=final_metrics["targets"],
        prob_scores=final_metrics["prob_scores"]
    )

    # delete final model checkpoint to save space
    if FINAL_MODEL_PATH.exists():
        os.remove(FINAL_MODEL_PATH)


    print(f"Finished run {run_idx+1}/{len(seeds)} | Seed={seed} | Best F1={best_metrics['f1']:.3f} | AUC={best_metrics['roc_auc']:.3f}\n")

print("\nAll runs completed successfully!")

