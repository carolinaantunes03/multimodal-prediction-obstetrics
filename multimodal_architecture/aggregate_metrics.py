import os
import json
import numpy as np
import pandas as pd
from pathlib import Path


base_results_dir = Path("results/exp_multi_seeds/exp3_FetalCLIP_ft/exp3_FetalCLIP_ft/cv3/")
output_xlsx = base_results_dir / "results_summary.xlsx"

metric_order = [
    "roc auc",
    "Recall",
    "Balanced Accuracy",
    "F1 Score",
    "Precision",
    "Matthews corrcoef",
    "Specificity",
    "NPV",
    "tp" ,
    "fn",
    "fp",
    "tn",
]


def fmt(x):
    if isinstance(x, (int, float)):
        return f"{x:.3f}".replace(".", ",")
    return x

# Collect all data 
rows = []
numeric_data_best = {m: [] for m in metric_order}
numeric_data_final = {m: [] for m in metric_order}

for seed_dir in sorted(base_results_dir.glob("cv3_seed*")):
    run_name = seed_dir.name

    best_path = seed_dir / "test_metrics_best.json"
    
    final_path = seed_dir / "test_metrics_final.json"

    if not best_path.exists() or not final_path.exists():
        continue

    with open(best_path, "r") as f:
        best_metrics = json.load(f)
    with open(final_path, "r") as f:
        final_metrics = json.load(f)

    # Collect numeric data for mean/std
    for m in metric_order:
        if m in best_metrics:
            numeric_data_best[m].append(best_metrics[m])
        if m in final_metrics:
            numeric_data_final[m].append(final_metrics[m])

    best_row = [fmt(best_metrics.get(m, "")) for m in metric_order]
    final_row = [fmt(final_metrics.get(m, "")) for m in metric_order]

    rows.append([run_name, "Best"] + best_row)
    rows.append([run_name, "Final"] + final_row)

# Convert to DataFrame 
columns = ["Model", "Type"] + metric_order
df = pd.DataFrame(rows, columns=columns)

# Compute Mean ± SD 
mean_row_best = ["Mean (Best)", ""] + [
    f"{np.mean(numeric_data_best[m]):.3f}".replace(".", ",") if len(numeric_data_best[m]) > 0 else ""
    for m in metric_order
]
std_row_best = ["SD (Best)", ""] + [
    f"{np.std(numeric_data_best[m]):.3f}".replace(".", ",") if len(numeric_data_best[m]) > 0 else ""
    for m in metric_order
]

mean_row_final = ["Mean (Final)", ""] + [
    f"{np.mean(numeric_data_final[m]):.3f}".replace(".", ",") if len(numeric_data_final[m]) > 0 else ""
    for m in metric_order
]
std_row_final = ["SD (Final)", ""] + [
    f"{np.std(numeric_data_final[m]):.3f}".replace(".", ",") if len(numeric_data_final[m]) > 0 else ""
    for m in metric_order
]

#df_summary = pd.DataFrame([mean_row_best, std_row_best, mean_row_final, std_row_final], columns=columns)
df_summary = pd.DataFrame([mean_row_best, std_row_best], columns=columns)

# Write Excel file 
with pd.ExcelWriter(output_xlsx, engine="xlsxwriter") as writer:
    df.to_excel(writer, index=False, sheet_name="All Runs")
    df_summary.to_excel(writer, index=False, sheet_name="Summary")

print(f"Excel file saved to: {output_xlsx}")
