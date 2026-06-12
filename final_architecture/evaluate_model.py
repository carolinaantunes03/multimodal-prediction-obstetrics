
import os
import re
import numpy as np
import pandas as pd
import torch
import joblib
from sklearn.metrics import (
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    confusion_matrix,
    roc_auc_score
)
import json

from final_model import (load_multimodal_model,
                            load_image,
                            extract_image_embeddings,
                            get_tabular_features,
                            concatenate_embeddings,
                            
                        )


def calculate_metrics(y_test, y_pred, y_proba_1):
    balanced_accuracy = balanced_accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average="weighted", zero_division=1)
    recall = recall_score(y_test, y_pred, average="weighted")
    f1 = f1_score(y_test, y_pred, average="weighted")
    macro_f1 = f1_score(y_test, y_pred, average="macro")
    matthews = matthews_corrcoef(y_test, y_pred)

    conf_matrix = confusion_matrix(y_test, y_pred, labels=[1, 0])
    roc_auc = roc_auc_score(y_test, y_proba_1)

    tp, fn, fp, tn = conf_matrix.ravel()

    tpr = tp / (tp + fp) if (tp + fp) > 0 else 0
    tnr = tn / (tn + fn) if (tn + fn) > 0 else 0
    acc = (tp + tn) / (tp + tn + fp + fn)
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0

    return {
        "balanced_accuracy": balanced_accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "macro_f1": macro_f1,
        "roc_auc": roc_auc,
        "matthews": matthews,
        "tnr": tnr,
        "specificity": specificity,
        "tp": tp,
        "fn": fn,
        "fp": fp,
        "tn": tn,
    }


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

retrospective_dataset = "../datasets/retrospective-dataset/all_retrospective_data.csv"
prospective_dataset = "../datasets/prospective-dataset/all_prospective_data_copy.csv"

with open("utils/columns_config.json", "r") as f:
    col_config = json.load(f)
num_cols = col_config["num_cols"]
cat_cols = col_config["cat_cols"]
image_cols = col_config["image_cols"]

processo = "Processo"
label = "Class"

multimodal_model_path = "../multimodal_architecture/models_prospective/MedViT2_nopt/final_model.pth"

train_df = pd.read_csv (retrospective_dataset)
train_df = train_df.drop(columns=[processo, label], errors="ignore")


test_df = pd.read_csv("../datasets/prospective-dataset/all_prospective_data.csv")


root_dir = "../datasets/prospective-dataset"

#load models
pre_trained_model = load_multimodal_model (multimodal_model_path, num_cols, cat_cols)
pipeline_path = "utils/final_pipeline_edca.pkl"

pipeline = joblib.load(pipeline_path)


results = []
for index,row in test_df.iterrows():

    head_img_path = row[image_cols[1]]
    femur_img_path = row[image_cols[2]]
    abdomen_img_path = row[image_cols[0]]

    head_img, head_valid = load_image (head_img_path, root_dir)
    head_img=head_img.to(DEVICE)
    femur_img, femur_valid = load_image(femur_img_path, root_dir)
    femur_img=femur_img.to(DEVICE)
    abdomen_img, abdomen_valid = load_image(abdomen_img_path, root_dir)
    abdomen_img=abdomen_img.to(DEVICE)

   
    tabular_features = get_tabular_features (row, num_cols, cat_cols)
  


    head_emb = extract_image_embeddings (pre_trained_model,head_img, head_valid)
    femur_emb = extract_image_embeddings (pre_trained_model,femur_img, femur_valid)
    abdomen_emb = extract_image_embeddings (pre_trained_model,abdomen_img, abdomen_valid)

    embedding_vector = concatenate_embeddings (head_emb, femur_emb, abdomen_emb, tabular_features)

    head_cols    = [f"image_head_emb{i+1}"    for i in range(head_emb.shape[0])]
    abdomen_cols = [f"image_abdomen_emb{i+1}" for i in range(abdomen_emb.shape[0])]
    femur_cols   = [f"image_femur_emb{i+1}"   for i in range(femur_emb.shape[0])]
    tabular_cols = num_cols + cat_cols

    all_cols = head_cols + abdomen_cols + femur_cols + tabular_cols
    

    embedding_df = pd.DataFrame([embedding_vector],columns = all_cols)
    #print(len(all_cols), embedding_vector.shape[0]) 
    
    y_proba = pipeline.predict_proba(embedding_df)[:, 1]
    y_pred = (y_proba >= 0.5).astype(int)
    results.append({
        "Processo": row[processo],
        "y_true":row[label],
        "y_pred":int(y_pred[0]),
        "y_proba_1":float(y_proba[0]),
        "y_proba_0":float(1 - y_proba[0])
    })

results_df = pd.DataFrame(results)
results_df.to_csv("results/edca_selected_data/predictions.csv", index=False)
print("Saved predictions.csv")

metrics = calculate_metrics(
    results_df["y_true"].values,
    results_df["y_pred"].values,
    results_df["y_proba_1"].values
)

metrics = {k: (v.item() if isinstance(v, (np.integer, np.floating)) else v) for k, v in metrics.items()}

with open("results/edca_selected_data/metrics.json", "w") as f:
    json.dump(metrics, f, indent=4)
print("Saved metrics.json")
print(metrics)