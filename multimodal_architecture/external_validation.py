import os
import sys
import json
import random
import torch
import pandas as pd
from pathlib import Path
from torch.utils.data import DataLoader

import data
import models
import utils


# Retrospective Dataset to train multimodal model
train_data_path = "../datasets/retrospective-dataset/all_retrospective_data.csv" #Path to retrospective dataset

# Prospective Dataset to evaluate the final multimodal model
prospective_data_path = "../datasets/prospective-dataset/all_prospective_data.csv"  #Path to prospective dataset

experiment_base = "MedViTV2-nopt"   #Path to save results
base_dir = Path(__file__).parent.resolve()
model_dir   = base_dir / "models_prospective" / experiment_base
results_dir = base_dir / "results_prospective" / experiment_base

for p in [model_dir, results_dir]:
    p.mkdir(parents=True, exist_ok=True)

# Training hyperparameters 

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED = 43   #this seed corresponds to the best run of the multimodal model
EPOCHS = 100
BATCH_SIZE = 16
LEARNING_RATE = 3.7e-4
WEIGHT_DECAY = 1e-3

FINAL_MODEL_PATH = model_dir / "final_model.pth"
CONFIG_FILE_PATH = model_dir / "config_model.json"
HISTORY_FILE_PATH = results_dir / "train_history.json"
METRICS_FILE_PATH = results_dir / "test_metrics.json"
PREDICTIONS_FILE_PATH = results_dir / "predictions.csv"
CM_FILE_PATH = results_dir / "conf_matrix.png"
ROC_FILE_PATH = results_dir / "roc_curve.png"

random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False



with open("columns_config.json", "r") as f:
    col_config = json.load(f)
num_cols = col_config["num_cols"]
cat_cols = col_config["cat_cols"]
image_cols = col_config["image_cols"]
y_col = "Class"


train_data = pd.read_csv(train_data_path)
prospective_data = pd.read_csv(prospective_data_path)

# Keep Processo column for predictions
train_data = train_data.dropna(subset=[y_col])
prospective_data = prospective_data.dropna(subset=[y_col])

tab_preproc = data.TabularPreprocessor(num_cols, cat_cols)
tab_preproc.fit(train_data)
possible_roots = [
    "/home/beatrix/Documents/Carolina_A/tese-carolina/datasets/retrospective-dataset",
    "/home/carolantunes/datasets/retrospective-dataset",
]
image_root = None
for root in possible_roots:
    candidate = os.path.join(root, "images/Abdomen")
    if os.path.exists(candidate):
        image_root = root
        break
if image_root is None:
    raise FileNotFoundError(" Could not locate image root directory.")
else:
    print(f" Using image root: {image_root}")

prospective_roots = [
    
    "/home/carolantunes/datasets/prospective-dataset",
    "/home/beatrix/Documents/Carolina_A/tese-carolina/datasets/prospective-dataset/prospective-images",
]


prospective_root = None
for root in prospective_roots:
    if os.path.exists(root):
        prospective_root = root
        break
if prospective_root is None:
    print("Could not find prospective image root")
    prospective_root = image_root
else:
    print(f"Using prospective image root: {prospective_root}")


#  Create datasets and dataloaders

train_dataset = data.MultimodalDataset(train_data, tab_preproc, image_cols, y_col, train=True, root_dir=image_root)
prospective_dataset = data.MultimodalDataset(prospective_data, tab_preproc, image_cols, y_col, train=False, root_dir=prospective_root)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
prospective_loader = DataLoader(prospective_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)


# Build multimodal model 

num_numerical = len(num_cols)
num_categories = [train_data[col].nunique() + 1 for col in cat_cols]


multimodal_model = models.build_multimodal_model(
    num_numerical=num_numerical,
    num_categories=num_categories,
    tabular_token_dim=192,
    tabular_hidden_dim=192
).to(DEVICE)

optimizer = torch.optim.AdamW(multimodal_model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
criterion = torch.nn.CrossEntropyLoss()

# Save config
utils.save_config_file(CONFIG_FILE_PATH, "AdamW", LEARNING_RATE, BATCH_SIZE, EPOCHS)


# Train on retrospective dataset

train_results = utils.train_only(
    multimodal_model,
    train_loader,
    optimizer,
    criterion,
    DEVICE,
    FINAL_MODEL_PATH,
    EPOCHS
)

# Save training history
utils.save_continued_train_metrics(
    HISTORY_FILE_PATH,
    train_results["train_losses"],
    train_results["train_accuracies"],
    train_results["elapsed_time"]
)


# Evaluate on prospective dataset

print("[INFO] Evaluating on prospective dataset...")
multimodal_model.load_state_dict(torch.load(FINAL_MODEL_PATH, map_location=DEVICE))
metrics = utils.evaluate(multimodal_model, prospective_loader, DEVICE)


# Save metrics, plots, and predictions

utils.save_test_metrics(
    METRICS_FILE_PATH,
    balanced_accuracy=metrics["balanced_accuracy"],
    acc=metrics["acc"],
    precision=metrics["precision"],
    recall=metrics["recall"],
    f1=metrics["f1"],
    roc_auc=metrics["roc_auc"],
    matthews=metrics["matthews"],
    specificity=metrics["specificity"],
    tnr=metrics["tnr"]
)

# Confusion Matrix & ROC Curve
utils.plot_confusionMatrix(
    CM_FILE_PATH,
    metrics["confusion_matrix"],
    class_names=['Cesarean birth', 'Vaginal birth'],
    title_fig="Confusion Matrix (Prospective Evaluation)"
)
utils.plot_RocCurve(
    ROC_FILE_PATH,
    targets=metrics["targets"],
    prob_scores=metrics["prob_scores"]
)

# Save predictions.csv (with Processo column)

prospective_processos = prospective_data["Processo"].reset_index(drop=True)
pred_df = pd.DataFrame({
    "Processo": prospective_processos,
    "y_true": metrics["targets"],
    "y_pred": metrics["predictions"],
    "y_proba_0": [p[0] for p in metrics["prob_scores"]],
    "y_proba_1": [p[1] for p in metrics["prob_scores"]],
})
pred_df.to_csv(PREDICTIONS_FILE_PATH, index=False)
print(f"Predictions saved to {PREDICTIONS_FILE_PATH}")

print("\nTraining and evaluation completed successfully!")
print(f"Results folder: {results_dir}")
