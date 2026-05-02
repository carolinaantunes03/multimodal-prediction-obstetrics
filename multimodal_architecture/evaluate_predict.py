
import os
import torch
import pandas as pd
from pathlib import Path
import json
import data
import models
import utils


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

import torch, random, numpy as np
seed = 43
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
random.seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.use_deterministic_algorithms(True)
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"


base_dir = Path(__file__).parent.resolve()
models_root = base_dir / "models" / "exp_multi_seeds"
results_root = base_dir / "results" / "exp_multi_seeds"

# Experiments
experiments = {
    #"Swin": "exp2_Swin",
    #"FetalCLIP": "exp2_FetalCLIP",
    #"MedMamba_pt": "exp2_MedMamba_pt",
    #"MedSigLip": "exp2_MedSigLip",
    #"MedSigLip_ft": "MedSigLip_finetuning",
    #"MedViT_nopt": "exp2_MedViT-nopt",
    "MedViT_pt": "teste_medvit"
}

folds = ["cv1", "cv2", "cv3"]


with open("columns_config.json", "r") as f:
    col_config = json.load(f)
num_cols = col_config["num_cols"]
cat_cols = col_config["cat_cols"]
image_cols = col_config["image_cols"]
y_col = "Class"

# Detect image root (debuging)
possible_roots = [
    "/home/beatrix/Documents/Carolina_A/tese-carolina/datasets/retrospective-dataset",
    "/home/carolantunes/datasets/retrospective-dataset",
]
image_root = next((r for r in possible_roots if os.path.exists(os.path.join(r, "images/Abdomen"))), None)
if image_root is None:
    raise FileNotFoundError("Could not find images directory.")
else:
    print(f"Using image root: {image_root}")


for exp_name, exp_dir in experiments.items():
    for cv in folds:
        print(f"\n=== Evaluating {exp_name} | {cv} ===")

        # --- Paths ---
        model_dir = models_root / exp_dir / cv
        result_dir = results_root / exp_dir / cv
        test_csv = f"../datasets/retrospective-dataset/{cv}/all_{cv}_test.csv"
        train_csv = f"../datasets/retrospective-dataset/{cv}/all_{cv}_train.csv"
        val_csv = f"../datasets/retrospective-dataset/{cv}/all_{cv}_validation.csv"

    
        best_model_paths = sorted(model_dir.rglob("best_model.pth"))

        if len(best_model_paths) == 0:
            raise FileNotFoundError(f"No best_model.pth found under {model_dir}")

        best_model_path = best_model_paths[0]
        print(f"Found model: {best_model_path}")

        # Data 
        test_data = pd.read_csv(test_csv)
        train_data = pd.read_csv(train_csv)
        val_data = pd.read_csv(val_csv)


        test_processo = test_data["Processo"].reset_index(drop=True)

        # Drop processo column for model input
        test_data_noproc = test_data.drop("Processo", axis=1)
        train_data_noproc = train_data.drop("Processo", axis=1)
        val_data_noproc = val_data.drop("Processo", axis=1)

        tab_preproc = data.TabularPreprocessor(num_cols, cat_cols)
        tab_preproc.fit(train_data_noproc)


        test_dataset = data.MultimodalDataset(
            test_data_noproc, tab_preproc, image_cols, y_col,
            train=False, root_dir=image_root
        )
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=16, shuffle=False)


        # Model
        num_numerical = len(num_cols)
        num_categories = [train_data[col].nunique() + 1 for col in cat_cols]

        multimodal_model = models.build_multimodal_model(
            num_numerical=num_numerical,
            num_categories=num_categories,
            tabular_token_dim=192,
            tabular_hidden_dim=192,
        ).to(device)

        state_dict = torch.load(best_model_path, map_location=device)
        multimodal_model.load_state_dict(state_dict, strict=True)
       
        multimodal_model.eval()
       

        for n, m in multimodal_model.named_modules():
            if hasattr(m, "training") and m.training:
                print("Still training:", n)


        # Evaluate
        with torch.no_grad():
            metrics = utils.evaluate(multimodal_model, test_loader, device)
        #metrics = utils.evaluate(multimodal_model, test_loader, DEVICE)

        # Save predictions 
        pred_dir = result_dir / "predictions"
        pred_dir.mkdir(parents=True, exist_ok=True)

        pred_csv = pred_dir / f"preds_{exp_name}_{cv}.csv"
        utils.save_predictions(
            predictions_file_path=pred_csv,
            y_true=metrics["targets"],
            y_pred=metrics["predictions"],
            prob_scores=metrics["prob_scores"],
            processo=test_processo
        )


        print(f"Saved predictions to: {pred_csv}")
