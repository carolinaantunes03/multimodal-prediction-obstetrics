import os
import torch
import json
import pandas as pd
import numpy as np
import random
from pathlib import Path
from torch.utils.data import DataLoader
import models_image
import data_image
import utils


SEEDS = [1, 6, 21, 10, 43, 50, 100, 336, 2025, 9999]
CVS   = ["cv1", "cv2", "cv3"]



test_encoder = ["MedSigLiP_finetune"]


encoders_config = {
    "Swin":       {"pretrained": True,  "freeze": False},
    "MedViT_pt":     {"pretrained_path": "../multimodal/MedViT_base_Fetal.pth", "freeze": False, "model_variant": "base"},
    "MedViT_nopt":     {"pretrained_path": None, "freeze": False, "model_variant": "base"},
    "MedMamba":   {"pretrained": True,  "pretrained_path": "../multimodal/MedMamba.pth",          "freeze": False, "model_variant": "s"},
    "FetalCLIP_2":  {"pretrained": True,  "freeze": True,"pretrained_path": "../multimodal/FetalCLIP_weights.pt"},
    "MedSigLIP_frozen":  {"pretrained": True,  "freeze": False},
    "MedSigLIP_finetune":  {"pretrained": True,  "freeze": True},
}

# Hyperparameters (same as multimodal)
epochs = 100
batch_size = 16
learning_rate = 3.7e-4
weight_decay = 1e-3
patience = 15


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


if __name__ == "__main__":

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load column config
    with open("columns_config.json", "r") as f:
        col_config = json.load(f)
    image_cols = col_config["image_cols"]
    y_col = "Class"

    # Detect dataset root
    possible_roots = [  #path to image data
        "/home/beatrix/Documents/Carolina_A/tese-carolina/datasets/retrospective-dataset",
        "/home/carolantunes/datasets/retrospective-dataset",
    ]
    image_root = next((r for r in possible_roots if os.path.exists(os.path.join(r, "images/Abdomen"))), None)
    if not image_root:
        raise FileNotFoundError("Could not find images directory.")
    print(f"Using image root: {image_root}")

    base_dir = Path(__file__).parent.resolve()

    # Loop over all encoders, CVs, and seeds
    for encoder_name in test_encoder:
      

        for cv in CVS:
            for seed in SEEDS:
                set_seed(seed)
                print(f"\n----- {encoder_name} | {cv} | Seed {seed} -----")

                # Paths
                train_path = f"../datasets/retrospective-dataset/{cv}/all_{cv}_train.csv"
                val_path   = f"../datasets/retrospective-dataset/{cv}/all_{cv}_validation.csv"
                test_path  = f"../datasets/retrospective-dataset/{cv}/all_{cv}_test.csv"

                train_df, val_df, test_df = (
                    pd.read_csv(train_path),
                    pd.read_csv(val_path),
                    pd.read_csv(test_path),
                )
                for df in [train_df, val_df, test_df]:
                    if "Processo" in df.columns:
                        df.drop("Processo", axis=1, inplace=True)

                # Datasets & loaders
                train_ds = data_image.ImageOnlyDataset(train_df, image_cols, y_col, image_root, train=True)
                val_ds   = data_image.ImageOnlyDataset(val_df, image_cols, y_col, image_root, train=False)
                test_ds  = data_image.ImageOnlyDataset(test_df, image_cols, y_col, image_root, train=False)

                train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
                val_loader   = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
                test_loader  = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

                # Debug 
                sample = next(iter(train_loader))
                print("Images shape:", sample["images"].shape)
                print("Valid counts:", sample["image_valid_num"])
                print("Labels:", sample["label"][:10])
                print("Min/Max pixel values:", sample["images"].min().item(), sample["images"].max().item())


                # Model setup
                cfg = encoders_config.get(encoder_name, {})
                model = models_image.build_image_model(encoder_name,num_classes=2,device=device,**cfg).to(device)
                #optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
                # Separate parameter groups
                
                backbone_params = []
                head_params = []

                for name, p in model.named_parameters():
                    if p.requires_grad:
                        if "head" in name or "proj" in name:
                            head_params.append(p)
                        else:
                            backbone_params.append(p)

                optimizer = torch.optim.AdamW([
                    {'params': backbone_params, 'lr': 1e-5},   # smaller LR for pre-trained layers
                    {'params': head_params, 'lr': 3e-4}        # larger LR for head
                ], weight_decay=weight_decay)

                n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
                print("Trainable params:", n_trainable)
                
                #criterion = torch.nn.CrossEntropyLoss()
                
               
                labels = [sample["label"] for sample in train_ds]
                class_counts = np.bincount(labels) 
                class_weights = 1. / torch.tensor(class_counts, dtype=torch.float)
                class_weights = class_weights / class_weights.sum() * len(class_counts)  # normalize
                criterion = torch.nn.CrossEntropyLoss(weight=class_weights.to(device))
            


                # Output paths
                run_name = f"{encoder_name}_imageonly_{cv}_seed{seed}"
                model_dir   = base_dir / "models" / run_name
                results_dir = base_dir / "results" / run_name
                model_dir.mkdir(parents=True, exist_ok=True)
                results_dir.mkdir(parents=True, exist_ok=True)

                best_model_path = model_dir / "best_model.pth"
                history_file = results_dir / "train_history.json"
                metrics_path = results_dir / "test_metrics_best.json"

                # Train and validate
                results = utils.train_and_validate(
                    model, train_loader, val_loader,
                    optimizer, criterion,
                    device, best_model_path,
                    epochs, patience
                )

                utils.save_train_metrics(
                    history_file,
                    results["train_losses"],
                    results["val_losses"],
                    results["train_accuracies"],
                    results["val_accuracies"],
                    results["best_epoch"],
                    results["elapsed_time"],
                )

                # Test best model
                cfg = encoders_config.get(encoder_name, {})
                best_model = models_image.build_image_model(encoder_name, num_classes=2, device=device,**cfg).to(device)
                best_model.load_state_dict(torch.load(best_model_path))
                best_metrics = utils.evaluate(best_model, test_loader, device)

                utils.save_test_metrics(
                    metrics_path,
                    best_metrics["balanced_accuracy"],
                    best_metrics["acc"],
                    best_metrics["precision"],
                    best_metrics["recall"],
                    best_metrics["f1"],
                    best_metrics["roc_auc"],
                    best_metrics["matthews"],
                    best_metrics["specificity"],
                    best_metrics["tnr"],
                )

                cm_path = results_dir / "conf_matrix_best.png"
                roc_curve_path = results_dir / "roc_curve_best.png"
                utils.plot_confusionMatrix(
                    cm_path,
                    best_metrics["confusion_matrix"],
                    ["Cesarean birth", "Vaginal birth"],
                    "Confusion Matrix (Best)"
                )
                utils.plot_RocCurve(
                    roc_curve_path,
                    best_metrics["targets"],
                    best_metrics["prob_scores"]
                )

                print(f"Done {encoder_name} | {cv} | Seed {seed} | "
                      f"F1={best_metrics['f1']:.3f} | AUC={best_metrics['roc_auc']:.3f}")

    print("\nAll image-only experiments completed successfully!")
