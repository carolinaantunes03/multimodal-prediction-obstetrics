import os
import json
import numpy as np
import pandas as pd
from sklearn.metrics import (
    roc_auc_score, recall_score, balanced_accuracy_score, f1_score,
    precision_score, matthews_corrcoef, confusion_matrix
)
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.metrics import cohen_kappa_score


model_name = "MedViT2_nopt_2"

model_preds_path = f"results_prospective/MedViT2_nopt/predictions.csv"
human_preds_path = "results_prospective/predictions_clinical_PC.csv"

output_root = Path(f"results_ensemble_human_model/{model_name}")
output_root.mkdir(parents=True, exist_ok=True)

voting_methods = ["mean_voting", "max_voting"]


def compute_metrics(y_true, y_pred, y_proba):
    cm = confusion_matrix(y_true, y_pred, labels=[1, 0])
    tp, fn, fp, tn = cm.ravel()

    roc_auc = roc_auc_score(y_true, y_proba)
    balanced_acc = balanced_accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='weighted', zero_division=1)
    recall = recall_score(y_true, y_pred, average='weighted')
    f1 = f1_score(y_true, y_pred, average='weighted')
    matthews = matthews_corrcoef(y_true, y_pred)

    tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
    tnr = tn / (tn + fp) if (tn + fp) > 0 else 0
    b_acc = (tpr + tnr) / 2
    acc = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0
    specificity = tnr

    metrics = {
        "balanced_accuracy": balanced_acc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "matthews": matthews,
        "roc_auc": roc_auc,
        "confusion_matrix": cm.tolist(),
        "tp": int(tp),
        "fn": int(fn),
        "fp": int(fp),
        "tn": int(tn),
        "tpr": float(tpr),
        "tnr": float(tnr),
        "b_acc": float(b_acc),
        "acc": float(acc),
        "specificity": float(specificity),
    }

    return metrics, cm

def plot_confusion_matrix(cm, save_path, title="Confusion Matrix"):
    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=["CS", "VD"], yticklabels=["CS", "VD"]
    )
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


df_model = pd.read_csv(model_preds_path)
df_human = pd.read_csv(human_preds_path)


df_merged = pd.merge(
    df_model,
    df_human[["Processo", "y_proba_1"]],
    on="Processo",
    how="inner",
    suffixes=("", "_human")
)


# Identify columns
col_pred = "y_pred" if "y_pred" in df_merged.columns else "y_pred_model"
col_proba = "y_proba_1" if "y_proba_1" in df_merged.columns else "y_proba_1_model"
col_proba_human = "y_proba_1_human"


df_merged = df_merged.copy()
df_merged["p_model"] = df_merged[col_proba].astype(float)
df_merged["p_human"] = df_merged[col_proba_human].astype(float)

df_merged["y_pred_model"] = df_merged[col_pred].astype(int)
df_merged["y_pred_human"] = (df_merged["p_human"] >= 0.5).astype(int)

# Mean voting (probability averaging)
df_merged["p_ensemble_mean"] = np.mean([df_merged["p_model"], df_merged["p_human"]], axis=0)
df_merged["y_pred_ensemble_mean"] = (df_merged["p_ensemble_mean"] >= 0.5).astype(int)

# Confidence = probability of predicted class
df_merged["model_confidence"] = np.where(
    df_merged["y_pred_model"] == 1,
    df_merged["p_model"],
    1 - df_merged["p_model"]
)

df_merged["human_confidence"] = np.where(
    df_merged["y_pred_human"] == 1,
    df_merged["p_human"],
    1 - df_merged["p_human"]
)

# Margins
df_merged["model_margin"] = np.abs(df_merged["p_model"] - 0.5)
df_merged["human_margin"] = np.abs(df_merged["p_human"] - 0.5)
df_merged["ensemble_margin"] = np.abs(df_merged["p_ensemble_mean"] - 0.5)

df_merged["disagreement_strength"] = np.abs(df_merged["p_model"] - df_merged["p_human"])

# Correctness flags
df_merged["correct_model"] = df_merged["y_pred_model"] == df_merged["y_true"]
df_merged["correct_human"] = df_merged["y_pred_human"] == df_merged["y_true"]

# Groups
df_merged["group"] = "both_wrong"
df_merged.loc[df_merged["correct_model"] & df_merged["correct_human"], "group"] = "both_correct"
df_merged.loc[df_merged["correct_model"] & ~df_merged["correct_human"], "group"] = "model_only_correct"
df_merged.loc[~df_merged["correct_model"] & df_merged["correct_human"], "group"] = "human_only_correct"


# ============================================================
for method in voting_methods:
    

    y_true = df_merged["y_true"].values
    proba_model = df_merged["p_model"].values
    proba_human = df_merged["p_human"].values

    preds_model = df_merged["y_pred_model"].values
    preds_human = df_merged["y_pred_human"].values

    if method == "mean_voting":
        y_proba_ensemble = np.mean([proba_model, proba_human], axis=0)
        y_pred_ensemble = (y_proba_ensemble >= 0.5).astype(int)

    elif method == "max_voting":
        preds_stack = np.stack([preds_model, preds_human], axis=1)
        y_pred_ensemble = np.array([np.bincount(row).argmax() for row in preds_stack])
        y_proba_ensemble = np.mean([proba_model, proba_human], axis=0)

    metrics, cm = compute_metrics(y_true, y_pred_ensemble, y_proba_ensemble)

    save_dir = output_root / method
    save_dir.mkdir(parents=True, exist_ok=True)

    df_out = pd.DataFrame({
        "Processo": df_merged["Processo"],
        "y_true": y_true,
        "y_pred": y_pred_ensemble,
        "y_proba_model": proba_model,
        "y_proba_human": proba_human,
        "y_proba_ensemble": y_proba_ensemble
    })
    df_out.to_csv(save_dir / "predictions_human_model_ensemble.csv", index=False)

    with open(save_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=4)

    pd.DataFrame([metrics]).to_excel(save_dir / "results_summary.xlsx", index=False)

    plot_confusion_matrix(cm, save_dir / "confusion_matrix.png",
                          title=f"Confusion Matrix ({method})")

    print(f"Ensemble {method} salvo em {save_dir}")

print("\nExperiência de ensemble humano + modelo concluída.")

# Complementary counts and plots

both_correct = np.sum(df_merged["group"] == "both_correct")
both_wrong = np.sum(df_merged["group"] == "both_wrong")
model_only = np.sum(df_merged["group"] == "model_only_correct")
human_only = np.sum(df_merged["group"] == "human_only_correct")

complementarity_data = pd.DataFrame({
    "Category": ["Both correct", "Both wrong", "Model only correct", "Human only correct"],
    "Count": [both_correct, both_wrong, model_only, human_only]
})

print("\nComplementarity analysis:")
print(complementarity_data)


plt.figure(figsize=(6, 5))
sns.barplot(data=complementarity_data, x="Category", y="Count", hue="Category", legend=False)
plt.title("Model vs Human Prediction Complementarity")
plt.ylabel("Number of Cases")
plt.xlabel("")
plt.xticks(rotation=15, ha="right")
plt.tight_layout()
plt.savefig(output_root / "complementarity_barplot.png")
plt.close()

complementarity_data["Percentage"] = 100 * complementarity_data["Count"] / len(df_merged)

plt.figure(figsize=(6, 5))
sns.barplot(data=complementarity_data, x="Category", y="Percentage", hue="Category", legend=False)
plt.title("Complementarity (%)")
plt.ylabel("Percentage (%)")
plt.xlabel("")
plt.xticks(rotation=15, ha="right")
plt.tight_layout()
plt.savefig(output_root / "complementarity_barplot_percentage.png")
plt.close()


groups_dir = output_root / "complementarity_groups"
groups_dir.mkdir(parents=True, exist_ok=True)

cols_export = [
    "Processo", "y_true",
    "y_pred_model", "p_model",
    "y_pred_human", "p_human",
    "y_pred_ensemble_mean", "p_ensemble_mean",
    "model_confidence", "human_confidence",
    "model_margin", "human_margin", "ensemble_margin",
    "disagreement_strength",
    "group"
]

df_merged[cols_export].to_csv(groups_dir / "all_cases_with_groups.csv", index=False)

for gname in ["both_correct", "both_wrong", "model_only_correct", "human_only_correct"]:
    df_subset = df_merged[df_merged["group"] == gname][cols_export]
    df_subset.to_csv(groups_dir / f"{gname}.csv", index=False)

print(f"CSVs por grupo guardados em: {groups_dir}")


# confidence plots

plots_dir = output_root / "confidence_plots"
plots_dir.mkdir(parents=True, exist_ok=True)

plt.figure(figsize=(8, 4))
sns.boxplot(data=df_merged, x="group", y="model_confidence")
plt.xticks(rotation=15, ha="right")
plt.title("Model confidence by group")
plt.tight_layout()
plt.savefig(plots_dir / "box_model_confidence_by_group.png")
plt.close()

plt.figure(figsize=(8, 4))
sns.boxplot(data=df_merged, x="group", y="human_confidence")
plt.xticks(rotation=15, ha="right")
plt.title("Human confidence by group")
plt.tight_layout()
plt.savefig(plots_dir / "box_human_confidence_by_group.png")
plt.close()

plt.figure(figsize=(8, 5))
for gname in ["both_correct", "both_wrong", "model_only_correct", "human_only_correct"]:
    sns.kdeplot(df_merged[df_merged["group"] == gname]["p_model"], label=gname)
plt.title("Model probability density by group")
plt.xlabel("p_model")
plt.legend()
plt.tight_layout()
plt.savefig(plots_dir / "kde_p_model_by_group.png")
plt.close()

print(f"Plots de confiança guardados em: {plots_dir}")


# COHEN'S KAPPA teste

kappa_model_human = cohen_kappa_score(df_merged["y_pred_model"], df_merged["y_pred_human"])
kappa_model_ens = cohen_kappa_score(df_merged["y_pred_model"], df_merged["y_pred_ensemble_mean"])
kappa_human_ens = cohen_kappa_score(df_merged["y_pred_human"], df_merged["y_pred_ensemble_mean"])

kappa_out = {
    "kappa_model_vs_human": float(kappa_model_human),
    "kappa_model_vs_ensemble_mean": float(kappa_model_ens),
    "kappa_human_vs_ensemble_mean": float(kappa_human_ens),
}

with open(output_root / "cohen_kappa.json", "w") as f:
    json.dump(kappa_out, f, indent=4)

print("\nCohen's Kappa results:")
print(json.dumps(kappa_out, indent=4))
print(f"Kappa guardado em: {output_root / 'cohen_kappa.json'}")

# Create a subdataset with the samples that only the model predict right 
prospective_full_path = "../datasets/prospective-dataset/all_prospective_data.csv"
df_full = pd.read_csv(prospective_full_path)

df_model_only = df_merged[df_merged["group"] == "model_only_correct"].copy()
processos_model_only = df_model_only["Processo"].astype(str).unique()

df_full["Processo"] = df_full["Processo"].astype(str)
df_full_model_only = df_full[df_full["Processo"].isin(processos_model_only)].copy()

subdataset_dir = output_root / "subdatasets"
subdataset_dir.mkdir(parents=True, exist_ok=True)

df_full_model_only.to_csv(subdataset_dir / "prospective_model_only_correct_fullvars.csv", index=False)
df_model_only[cols_export].to_csv(subdataset_dir / "prospective_model_only_correct_predictions.csv", index=False)

print(f"\nSubdataset isolado (model_only_correct) guardado em: {subdataset_dir}")


# class distribution in the samples that only the model predict right 
if "Class" in df_full_model_only.columns:
    counts = df_full_model_only["Class"].value_counts(dropna=False).to_dict()
    total = len(df_full_model_only)

    n_cs = counts.get(1, 0)
    n_vd = counts.get(0, 0)

    summary = {
        "total_cases_model_only_correct": int(total),
        "n_CS": int(n_cs),
        "n_VD": int(n_vd),
        "pct_CS": float(n_cs / total) if total > 0 else 0.0,
        "pct_VD": float(n_vd / total) if total > 0 else 0.0,
    }

    with open(subdataset_dir / "model_only_correct_class_distribution.json", "w") as f:
        json.dump(summary, f, indent=4)

    print("\n Class distribution in model_only_correct subset:")
    print(json.dumps(summary, indent=4))
    print(f"Guardado em: {subdataset_dir / 'model_only_correct_class_distribution.json'}")
else:
    print("Coluna 'Class' não encontrada no all_prospective_data.csv.")
