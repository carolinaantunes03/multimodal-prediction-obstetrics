import os
import sys
import json
import random
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from torch.utils.data import DataLoader

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '')))
import data
import models
import utils


base_dataset_dir = "../datasets/retrospective-dataset" #path to retrospective dataset
experiment_base = "tabular_only/FTTransformer"  #path to save results
folds = [1, 2, 3]
seeds = [1, 6, 21, 10, 43, 50, 100, 336, 2025, 9999]

epochs = 100
patience = 15
batch_size = 16
learning_rate = 3.7e-4
weight_decay = 1e-3
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"Using device: {device}")


with open("columns_config.json", "r") as f:
    col_config = json.load(f)
num_cols = col_config["num_cols"]
cat_cols = col_config["cat_cols"]
y_col = "Class"


for cv_fold in folds:
    print(f"\n{'='*60}\nStarting Fold {cv_fold}\n{'='*60}")

    # Load data
    train_path = f"{base_dataset_dir}/cv{cv_fold}/all_cv{cv_fold}_train.csv"
    val_path   = f"{base_dataset_dir}/cv{cv_fold}/all_cv{cv_fold}_validation.csv"
    test_path  = f"{base_dataset_dir}/cv{cv_fold}/all_cv{cv_fold}_test.csv"

    train_df = pd.read_csv(train_path).drop(columns=["Processo"], errors="ignore")
    val_df   = pd.read_csv(val_path).drop(columns=["Processo"], errors="ignore")
    test_df  = pd.read_csv(test_path).drop(columns=["Processo"], errors="ignore")

    # Fit tabular preprocessor
    tab_preproc = data.TabularPreprocessor(num_cols, cat_cols)
    tab_preproc.fit(train_df)

    # Build datasets (pure tabular, no images)
    train_dataset = data.TabularDataset(train_df, tab_preproc, y_col)
    val_dataset   = data.TabularDataset(val_df, tab_preproc, y_col)
    test_dataset  = data.TabularDataset(test_df, tab_preproc, y_col)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader  = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    num_numerical = len(num_cols)
    num_categories = [train_df[col].nunique() + 1 for col in cat_cols]


    for seed in seeds:
        print(f"\n{'-'*40}\nFold {cv_fold} | Seed {seed}\n{'-'*40}")

        # Set seeds
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        # Paths
        base_dir = Path(__file__).parent.resolve()
        run_name = f"FTTransformer_cv{cv_fold}_seed{seed}"
        model_dir   = base_dir / "models" / experiment_base / f"cv{cv_fold}" / f"seed{seed}"
        results_dir = base_dir / "results" / experiment_base / f"cv{cv_fold}" / f"seed{seed}"
        for p in [model_dir, results_dir]:
            p.mkdir(parents=True, exist_ok=True)

        best_model_path  = model_dir / "best_model.pth"
        final_model_path = model_dir / "final_model.pth"
        history_file = results_dir / "train_history.json"
        config_file  = model_dir / "config_model.json"
        metrics_best_model  = results_dir / "test_metrics_best.json"
        predictions_file = results_dir / "preds_FTTransformer_cv{}_seed{}.csv".format(cv_fold, seed)

    
        model = models.FTTransformer(
            num_numerical=num_numerical,
            num_categories=num_categories,
            token_dim=192,
            hidden_size=192,
            num_blocks=3,
            attention_n_heads=8,
            num_classes=2
        ).to(device)

        optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        criterion = torch.nn.CrossEntropyLoss()

        # Save config
        utils.save_config_file(config_file, "AdamW", learning_rate, batch_size, epochs)

       
        results = utils.train_and_validate(
            model, train_loader, val_loader, optimizer, criterion,
            device, best_model_path, final_model_path, epochs, patience
        )

        utils.save_train_metrics(
            history_file_path=history_file,
            train_losses=results["train_losses"],
            val_losses=results["val_losses"],
            train_accuracies=results["train_accuracies"],
            val_accuracies=results["val_accuracies"],
            best_epoch=results["best_epoch"],
            elapsed_time=results["elapsed_time"]
        )

        
        best_model = models.FTTransformer(
            num_numerical=num_numerical,
            num_categories=num_categories,
            token_dim=192,
            hidden_size=192,
            num_blocks=3,
            attention_n_heads=8,
            num_classes=2
        ).to(device)

        best_model.load_state_dict(torch.load(best_model_path))
        test_metrics = utils.evaluate(best_model, test_loader, device)

        utils.save_test_metrics(
            metrics_file_path=metrics_best_model,
            balanced_accuracy=test_metrics["balanced_accuracy"],
            acc=test_metrics["acc"],
            precision=test_metrics["precision"],
            recall=test_metrics["recall"],
            f1=test_metrics["f1"],
            roc_auc=test_metrics["roc_auc"],
            matthews=test_metrics["matthews"],
            specificity=test_metrics["specificity"],
            tnr=test_metrics["tnr"]
        )

        # Save predictions (y_true, y_pred, y_proba_0, y_proba_1)
        utils.save_predictions(
            predictions_file,
            y_true=test_metrics["targets"],
            y_pred=test_metrics["predictions"],
            prob_scores=test_metrics["prob_scores"]
        )

        print(f"Finished Fold {cv_fold} | Seed {seed} | F1={test_metrics['f1']:.3f} | AUC={test_metrics['roc_auc']:.3f}")
