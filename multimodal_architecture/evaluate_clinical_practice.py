import pandas as pd
import numpy as np
from sklearn.metrics import (
    confusion_matrix,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    roc_auc_score
)
import torch.nn.functional as F  # para manter consistência
import json


csv_path = "../datasets/prospective-dataset/PT_external_dataset_processed.csv" #Path to clinical practice file
excel_path = "../datasets/prospective-dataset/Base_dados_prospetiva.xlsx"   

df_main = pd.read_csv(csv_path)
df_pc = pd.read_excel(excel_path)

df_main.columns = df_main.columns.str.strip()
df_pc.columns = df_pc.columns.str.strip()

# Agreggate csv file with prospective data with clinical practice using "Processo"

df = pd.merge(df_main, df_pc[["Processo", "PC"]], on="Processo", how="inner")
print(f"Total samples: {len(df)}")


# Normalize Clinical Practice [0,1]

min_pc = df["PC"].min()
max_pc = df["PC"].max()
df["PC_normalized"] = (df["PC"] - min_pc) / (max_pc - min_pc)
print(f"Intervalo original PC: {min_pc} → {max_pc}")


y_true = df["Class"].astype(int).to_numpy()
y_score = df["PC_normalized"].to_numpy()
y_pred = (y_score > 0.5).astype(int)  #CHANGE HERE YOU WANT > 0.5 OR >= 0.5 

balanced_accuracy = balanced_accuracy_score(y_true, y_pred)
precision = precision_score(y_true, y_pred, average="weighted", zero_division=1)
recall = recall_score(y_true, y_pred, average="weighted")
f1 = f1_score(y_true, y_pred, average="weighted")
matthews = matthews_corrcoef(y_true, y_pred)
test_conf_matrix = confusion_matrix(y_true, y_pred, labels=[1, 0])
roc_auc = roc_auc_score(y_true, y_score)

tp, fn, fp, tn = test_conf_matrix.ravel()
tpr = tp / (tp + fp) if (tp + fp) > 0 else 0
tnr = tn / (tn + fn) if (tn + fn) > 0 else 0
b_acc = (tpr + tnr) / 2
acc = (tp + tn) / (tp + tn + fp + fn)
specificity = tn / (tn + fp) if (tn + fp) > 0 else 0


print("\n===== Métricas Previsão Clínica (compatível com multimodal) =====")
print(f"Balanced Accuracy: {balanced_accuracy:.4f} | Precision: {precision:.4f} | Recall: {recall:.4f} | F1: {f1:.4f}")
print(f"Matthews Corrcoef: {matthews:.4f} | ROC AUC: {roc_auc:.4f}")
print(f"Confusion Matrix:\n{test_conf_matrix}")
print(f"TP={tp}, FN={fn}, FP={fp}, TN={tn}")
print(f"Specificity={specificity:.4f}, tpr={tpr:.4f}, tnr={tnr:.4f}, b_acc={b_acc:.4f}, acc={acc:.4f}")


predictions_file = "results/predictions_clinical_PC.csv"
df_preds = pd.DataFrame({
    "Processo": df["Processo"],
    "y_true": y_true,
    "y_pred": y_pred,
    "y_proba": y_score
})
df_preds.to_csv(predictions_file, index=False)
print(f"\nFicheiro de previsões guardado em: {predictions_file}")


metrics_file = "results/metrics_clinical_PC.json"
metrics = {
    "balanced_accuracy": balanced_accuracy,
    "precision": precision,
    "recall": recall,
    "f1": f1,
    "matthews": matthews,
    "roc_auc": roc_auc,
    "tp": int(tp),
    "fn": int(fn),
    "fp": int(fp),
    "tn": int(tn),
    "tpr": tpr,
    "tnr": tnr,
    "b_acc": b_acc,
    "acc": acc,
    "specificity": specificity
}

with open(metrics_file, "w") as f:
    json.dump(metrics, f, indent=4)

print(f"Ficheiro de métricas guardado em: {metrics_file}")