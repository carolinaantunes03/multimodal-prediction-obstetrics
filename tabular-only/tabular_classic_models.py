# tabular_classic_ml.py
import os
import json
import random
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    matthews_corrcoef,
)


base_dataset_dir = "../datasets/retrospective-dataset" #path to retrospective dataset
experiment_base = "tabular_only/classic_ml" #path to save results
folds = [1, 2, 3]
seeds = [1, 6, 21, 10, 43, 50, 100, 336, 2025, 9999]


with open("columns_config.json", "r") as f:
    col_config = json.load(f)

num_cols = col_config["num_cols"]
cat_cols = col_config["cat_cols"]
y_col = "Class"

# loop over cv folds

for cv_fold in folds:
    print(f"\n{'='*60}\nStarting Fold {cv_fold}\n{'='*60}")

    # Load data
    train_path = f"{base_dataset_dir}/cv{cv_fold}/all_cv{cv_fold}_train.csv"
    val_path   = f"{base_dataset_dir}/cv{cv_fold}/all_cv{cv_fold}_validation.csv"
    test_path  = f"{base_dataset_dir}/cv{cv_fold}/all_cv{cv_fold}_test.csv"

    train_df = pd.read_csv(train_path).drop(columns=["Processo"], errors="ignore")
    val_df   = pd.read_csv(val_path).drop(columns=["Processo"], errors="ignore")
    test_df  = pd.read_csv(test_path).drop(columns=["Processo"], errors="ignore")

  
    for seed in seeds:
        print(f"\n{'-'*40}\nFold {cv_fold} | Seed {seed}\n{'-'*40}")
        random.seed(seed)
        np.random.seed(seed)

        
        preprocessor = ColumnTransformer(
            transformers=[
                ("num", StandardScaler(), num_cols),
                ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols)
            ]
        )

        X_train = train_df.drop(columns=[y_col])
        y_train = train_df[y_col]
        X_val   = val_df.drop(columns=[y_col])
        y_val   = val_df[y_col]
        X_test  = test_df.drop(columns=[y_col])
        y_test  = test_df[y_col]

        # Define classic ml models
         
        models_dict = {
            "LogisticRegression": LogisticRegression(max_iter=1000, random_state=seed),
            "RandomForest": RandomForestClassifier(n_estimators=200, max_depth=None, random_state=seed),
            "AdaBoost": AdaBoostClassifier(n_estimators=200, random_state=seed),
            "XGBoost": XGBClassifier(n_estimators=200, use_label_encoder=False, eval_metric="logloss", random_state=seed),
        }

        for model_name, model in models_dict.items():
            pipeline = Pipeline(steps=[("preprocessor", preprocessor), ("classifier", model)])
            pipeline.fit(X_train, y_train)

            y_pred = pipeline.predict(X_test)
            y_proba = pipeline.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None

            
            test_conf_matrix = confusion_matrix(y_test, y_pred, labels=[1, 0])
            tp, fn, fp, tn = test_conf_matrix.ravel()
            balanced_acc = balanced_accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred, zero_division=1, average='weighted')
            recall = recall_score(y_test, y_pred, average='weighted')
            f1 = f1_score(y_test, y_pred, average='weighted')
            matthews = matthews_corrcoef(y_test, y_pred)
            roc_auc = roc_auc_score(y_test, y_proba) if y_proba is not None else None
            specificity = tn / (tn + fp)

            
            base_dir = Path(__file__).parent.resolve()
            results_dir = base_dir / "results" / experiment_base / f"cv{cv_fold}" / f"seed{seed}" / model_name
            results_dir.mkdir(parents=True, exist_ok=True)

            metrics_file = results_dir / "metrics.json"
            results_df = pd.DataFrame({
                "y_true": y_test,
                "y_pred": y_pred,
                "y_proba": y_proba if y_proba is not None else [None]*len(y_test)
            })
            results_df.to_csv(results_dir / "predictions.csv", index=False)

          
            metrics = {
                "balanced_accuracy": float(balanced_acc),
                "precision": float(precision),
                "recall": float(recall),
                "f1": float(f1),
                "matthews": float(matthews),
                "roc_auc": float(roc_auc) if roc_auc is not None else None,
                "specificity": float(specificity),
                "tp": int(tp),
                "tn": int(tn),
                "fp": int(fp),
                "fn": int(fn)
            }

            with open(metrics_file, "w") as f:
                json.dump(metrics, f, indent=2)

            print(f"{model_name} | Fold {cv_fold} | Seed {seed} | F1={f1:.3f} | AUC={roc_auc:.3f}")

